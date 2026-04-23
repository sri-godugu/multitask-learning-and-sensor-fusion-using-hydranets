import os
import cv2
import numpy as np
import torch

PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32),
]


def seg_to_color(seg_map):
    if isinstance(seg_map, torch.Tensor):
        seg_map = seg_map.cpu().numpy()
    H, W = seg_map.shape
    color = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, rgb in enumerate(PALETTE):
        color[seg_map == cls_id] = rgb
    return color


def depth_to_color(depth):
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    d_min, d_max = depth.min(), depth.max()
    d_norm = ((depth - d_min) / (d_max - d_min + 1e-7) * 255).astype(np.uint8)
    return cv2.applyColorMap(d_norm, cv2.COLORMAP_MAGMA)


def visualize_outputs(orig_rgb, outputs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, 'input.jpg'),
                cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR))

    if 'segmentation' in outputs:
        seg_map = outputs['segmentation'][0].argmax(0)
        cv2.imwrite(os.path.join(save_dir, 'segmentation.png'),
                    seg_to_color(seg_map))

    if 'depth' in outputs:
        cv2.imwrite(os.path.join(save_dir, 'depth.png'),
                    depth_to_color(outputs['depth'][0]))
