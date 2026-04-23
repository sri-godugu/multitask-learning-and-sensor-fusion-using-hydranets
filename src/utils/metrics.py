import numpy as np
import torch


def mean_iou(pred, target, num_classes, ignore_index=255):
    """Mean IoU for semantic segmentation."""
    pred   = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    ious = []
    for cls in range(num_classes):
        tp = int(((pred == cls) & (target == cls)).sum())
        fp = int(((pred == cls) & (target != cls) & (target != ignore_index)).sum())
        fn = int(((pred != cls) & (target == cls)).sum())
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
    return float(np.mean(ious)) if ious else 0.0


def abs_rel_error(pred, target):
    """Absolute relative depth error — lower is better."""
    valid = target > 0
    if valid.sum() == 0:
        return float('nan')
    return ((pred[valid] - target[valid]).abs() / target[valid]).mean().item()


def rmse_depth(pred, target):
    """Root mean squared depth error — lower is better."""
    valid = target > 0
    if valid.sum() == 0:
        return float('nan')
    return ((pred[valid] - target[valid]) ** 2).mean().sqrt().item()


def delta_accuracy(pred, target, threshold=1.25):
    """% of pixels where max(pred/gt, gt/pred) < threshold — higher is better."""
    valid = target > 0
    if valid.sum() == 0:
        return float('nan')
    ratio = torch.max(pred[valid] / target[valid], target[valid] / pred[valid])
    return (ratio < threshold).float().mean().item()
