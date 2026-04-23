"""
Fine-tune only the depth estimation head on a pretrained HydraNet.

Three-phase freezing strategy:
  Phase 1 (epochs 1 → phase1_epochs):
      Backbone + FPN + all other heads frozen  →  only depth_head trains
  Phase 2 (phase1_epochs+1 → phase1+phase2_epochs):
      FPN unfrozen  →  depth_head + FPN train together
  Phase 3 (remainder):
      Full network unfrozen  →  end-to-end fine-tuning at low LR

Usage:
    python scripts/finetune_depth.py \\
        --checkpoint checkpoints/hydranet_epoch050.pt \\
        --data /path/to/depth_data
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
from src.losses.depth_loss import ScaleInvariantDepthLoss
from src.data.dataset import MultiTaskDataset
from src.data.transforms import build_train_transforms
from src.utils.freeze_utils import (
    freeze_backbone, freeze_all_heads, unfreeze_heads,
    unfreeze_module, unfreeze_backbone, print_trainable_summary,
)


def parse_args():
    p = argparse.ArgumentParser(description='Selective depth-head fine-tuning')
    p.add_argument('--checkpoint',    required=True)
    p.add_argument('--config',        default='configs/hydranet.yaml')
    p.add_argument('--data',          required=True)
    p.add_argument('--output',        default='checkpoints/depth_ft')
    p.add_argument('--epochs',        type=int,   default=20)
    p.add_argument('--phase1-epochs', type=int,   default=5)
    p.add_argument('--phase2-epochs', type=int,   default=10)
    p.add_argument('--batch-size',    type=int,   default=8)
    p.add_argument('--lr',            type=float, default=5e-5)
    p.add_argument('--device',        default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg['active_tasks'] = ['depth']   # only run depth head in forward pass
    os.makedirs(args.output, exist_ok=True)

    model = HydraNet(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get('model', state), strict=False)
    print(f"Loaded: {args.checkpoint}")

    # ── Phase 1: only depth_head trainable ────────────────────────────
    freeze_backbone(model)
    freeze_all_heads(model)
    unfreeze_heads(model, ['depth'])
    print("\nPhase 1 — depth head only:")
    print_trainable_summary(model)

    criterion = ScaleInvariantDepthLoss().to(device)
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    img_size = tuple(cfg.get('img_size', [512, 512]))
    ds = MultiTaskDataset(args.data, split='train', tasks=['depth'],
                          transforms=build_train_transforms(img_size))
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(1, args.epochs + 1):
        # ── Phase transitions ──────────────────────────────────────────
        if epoch == args.phase1_epochs + 1:
            unfreeze_module(model.fpn)
            # add new params to optimizer
            optimizer.add_param_group({'params': [p for p in model.fpn.parameters()
                                                   if p.requires_grad],
                                       'lr': args.lr * 0.5})
            print(f"\nPhase 2 (epoch {epoch}): FPN unfrozen")
            print_trainable_summary(model)

        if epoch == args.phase1_epochs + args.phase2_epochs + 1:
            unfreeze_backbone(model)
            optimizer.add_param_group({'params': [p for p in model.backbone.parameters()
                                                   if p.requires_grad],
                                       'lr': args.lr * 0.1})
            print(f"\nPhase 3 (epoch {epoch}): full network unfrozen")
            print_trainable_summary(model)

        model.train()
        running = 0.0
        for batch in loader:
            if 'depth' not in batch:
                continue
            rgb  = batch['image'].to(device)
            gt_d = batch['depth'].to(device)

            pred_d = model(rgb)['depth']
            loss   = criterion(pred_d, gt_d)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()

        avg = running / max(len(loader), 1)
        print(f"Epoch {epoch:3d}/{args.epochs} | depth loss: {avg:.4f}")
        torch.save({'epoch': epoch, 'model': model.state_dict()},
                   os.path.join(args.output, f'depth_ft_epoch{epoch:03d}.pt'))

    print("Depth fine-tuning complete.")


if __name__ == '__main__':
    main()
