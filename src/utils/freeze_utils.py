from __future__ import annotations
import torch.nn as nn


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def unfreeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True
    module.train()


def freeze_backbone(model: nn.Module):
    """Freeze shared backbone and FPN; task heads remain trainable."""
    freeze_module(model.backbone)
    freeze_module(model.fpn)


def unfreeze_backbone(model: nn.Module):
    unfreeze_module(model.backbone)
    unfreeze_module(model.fpn)


def freeze_all_heads(model: nn.Module):
    for attr in ('det_head', 'seg_head', 'loc_head', 'depth_head'):
        if hasattr(model, attr):
            freeze_module(getattr(model, attr))


def unfreeze_heads(model: nn.Module, names: list[str]):
    for name in names:
        attr = name if name.endswith('_head') else f'{name}_head'
        if hasattr(model, attr):
            unfreeze_module(getattr(model, attr))


def get_trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def print_trainable_summary(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:>12,} / {total:>12,}  ({100 * trainable / max(total, 1):.1f}%)")
    for name, child in model.named_children():
        ct = sum(p.numel() for p in child.parameters() if p.requires_grad)
        cn = sum(p.numel() for p in child.parameters())
        status = 'trainable' if ct > 0 else 'frozen   '
        print(f"  {name:<20s}: {ct:>10,} / {cn:>10,}  [{status}]")
