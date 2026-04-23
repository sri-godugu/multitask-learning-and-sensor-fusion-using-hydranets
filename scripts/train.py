"""
Train HydraNet end-to-end with uncertainty-weighted multi-task loss.

Usage:
    python scripts/train.py --config configs/hydranet.yaml --data /path/to/data
    python scripts/train.py --config configs/hydranet.yaml --data /path/to/data \\
        --freeze-backbone --epochs 50 --batch-size 8
"""
import argparse
import os
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.hydranet import HydraNet
from src.losses.multitask_loss import MultitaskLoss
from src.losses.detection_loss import FCOSLoss
from src.losses.segmentation_loss import SegmentationLoss
from src.losses.depth_loss import ScaleInvariantDepthLoss
from src.losses.localization_loss import LocalizationLoss
from src.data.dataset import MultiTaskDataset
from src.data.transforms import build_train_transforms, build_val_transforms
from src.utils.freeze_utils import freeze_backbone, unfreeze_backbone, print_trainable_summary


def parse_args():
    p = argparse.ArgumentParser(description='HydraNet multi-task training')
    p.add_argument('--config',    default='configs/hydranet.yaml')
    p.add_argument('--data',      required=True,  help='Root data directory')
    p.add_argument('--output',    default='checkpoints')
    p.add_argument('--epochs',    type=int,   default=50)
    p.add_argument('--batch-size',type=int,   default=8)
    p.add_argument('--lr',        type=float, default=1e-4)
    p.add_argument('--freeze-backbone', action='store_true',
                   help='Freeze backbone during warm-up phase')
    p.add_argument('--resume',    default=None, help='Path to checkpoint')
    p.add_argument('--device',    default=None)
    return p.parse_args()


def build_criterion(cfg):
    tasks = cfg.get('active_tasks', ['detection', 'segmentation', 'localization', 'depth'])
    fns = {
        'detection':    FCOSLoss(cfg.get('num_det_classes', 80)),
        'segmentation': SegmentationLoss(cfg.get('num_seg_classes', 19)),
        'localization': LocalizationLoss(),
        'depth':        ScaleInvariantDepthLoss(),
    }
    active = {k: v for k, v in fns.items() if k in tasks}
    return MultitaskLoss(list(active.keys()), active)


def main():
    args   = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output, exist_ok=True)
    img_size = tuple(cfg.get('img_size', [512, 512]))

    model = HydraNet(cfg).to(device)
    if args.freeze_backbone:
        freeze_backbone(model)
        print("Backbone frozen for warm-up.")
    print_trainable_summary(model)

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state.get('model', state), strict=False)
        print(f"Resumed from {args.resume}")

    criterion = build_criterion(cfg).to(device)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.log_vars.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_ds = MultiTaskDataset(args.data, split='train',
                                tasks=cfg.get('active_tasks'),
                                transforms=build_train_transforms(img_size))
    val_ds   = MultiTaskDataset(args.data, split='val',
                                tasks=cfg.get('active_tasks'),
                                transforms=build_val_transforms(img_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    warmup = cfg.get('backbone_warmup_epochs', 5)

    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone and epoch == warmup + 1:
            unfreeze_backbone(model)
            print(f"\nEpoch {epoch}: backbone unfrozen for full fine-tuning.")
            print_trainable_summary(model)

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            rgb       = batch['image'].to(device)
            secondary = batch['depth'].to(device) if 'depth' in batch else None
            targets   = {}
            if 'seg'   in batch: targets['segmentation'] = batch['seg'].to(device)
            if 'depth' in batch: targets['depth']        = batch['depth'].to(device)

            outputs = model(rgb, secondary)
            loss, loss_dict = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            if step % 50 == 0:
                parts = ' | '.join(f"{k}: {v:.4f}" for k, v in loss_dict.items()
                                   if k != 'total' and isinstance(v, float))
                print(f"\r  [{epoch}/{args.epochs}] step {step:4d} | {parts}", end='', flush=True)

        scheduler.step()
        print(f"\nEpoch {epoch} — avg train loss: {total_loss / max(len(train_loader),1):.4f}")

        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'cfg': cfg},
                   os.path.join(args.output, f'hydranet_epoch{epoch:03d}.pt'))

    print("Training complete.")


if __name__ == '__main__':
    main()
