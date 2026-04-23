"""
Evaluate a HydraNet checkpoint on segmentation (mIoU) and depth metrics.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/hydranet_epoch050.pt \\
        --data /path/to/data --split val
"""
import argparse
import os
import sys
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.hydranet import HydraNet
from src.data.dataset import MultiTaskDataset
from src.data.transforms import build_val_transforms
from src.utils.metrics import mean_iou, abs_rel_error, rmse_depth, delta_accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config',     default='configs/hydranet.yaml')
    p.add_argument('--data',       required=True)
    p.add_argument('--split',      default='val')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device',     default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = HydraNet(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get('model', state), strict=False)
    model.eval()

    img_size = tuple(cfg.get('img_size', [512, 512]))
    ds = MultiTaskDataset(args.data, split=args.split,
                          tasks=cfg.get('active_tasks'),
                          transforms=build_val_transforms(img_size))
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=4)

    seg_ious, abs_rels, rmses, deltas = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            rgb       = batch['image'].to(device)
            secondary = batch['depth'].to(device) if 'depth' in batch else None
            outputs   = model(rgb, secondary)

            if 'segmentation' in outputs and 'seg' in batch:
                pred = outputs['segmentation'].argmax(1)
                gt   = batch['seg'].to(device)
                pred = F.interpolate(pred.unsqueeze(1).float(),
                                     size=gt.shape[-2:], mode='nearest').squeeze(1).long()
                seg_ious.append(mean_iou(pred, gt, cfg.get('num_seg_classes', 19)))

            if 'depth' in outputs and 'depth' in batch:
                pd = outputs['depth'].to(device)
                gd = batch['depth'].to(device)
                abs_rels.append(abs_rel_error(pd, gd))
                rmses.append(rmse_depth(pd, gd))
                deltas.append(delta_accuracy(pd, gd))

    print("\n=== Evaluation Results ===")
    if seg_ious:
        print(f"Segmentation  mIoU : {sum(seg_ious)/len(seg_ious):.4f}")
    if abs_rels:
        print(f"Depth  AbsRel      : {sum(abs_rels)/len(abs_rels):.4f}")
        print(f"Depth  RMSE        : {sum(rmses)/len(rmses):.4f}")
        print(f"Depth  δ<1.25      : {sum(deltas)/len(deltas):.4f}")


if __name__ == '__main__':
    main()
