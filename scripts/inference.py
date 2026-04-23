"""
Run HydraNet on a single image and save per-task visualisations.

Usage:
    python scripts/inference.py --image path/to/image.jpg \\
        --checkpoint checkpoints/hydranet_epoch050.pt
    # with sensor fusion:
    python scripts/inference.py --image rgb.jpg --depth-input depth.png \\
        --checkpoint checkpoints/hydranet_epoch050.pt
"""
import argparse
import os
import sys
import yaml
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.hydranet import HydraNet
from src.utils.visualization import visualize_outputs


def parse_args():
    p = argparse.ArgumentParser(description='HydraNet single-image inference')
    p.add_argument('--image',       required=True)
    p.add_argument('--depth-input', default=None, help='Optional depth/LiDAR PNG')
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--config',      default='configs/hydranet.yaml')
    p.add_argument('--output',      default='output')
    p.add_argument('--img-size',    type=int, nargs=2, default=[512, 512], metavar=('H','W'))
    p.add_argument('--device',      default=None)
    return p.parse_args()


def preprocess(img_bgr, img_size):
    H, W = img_size
    img = cv2.cvtColor(cv2.resize(img_bgr, (W, H)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0), img


def main():
    args   = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model = HydraNet(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get('model', state), strict=False)
    model.eval()

    img_bgr  = cv2.imread(args.image)
    rgb_t, orig_rgb = preprocess(img_bgr, args.img_size)
    rgb_t = rgb_t.to(device)

    secondary = None
    if args.depth_input and os.path.exists(args.depth_input):
        d = cv2.imread(args.depth_input, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        d = cv2.resize(d, (args.img_size[1], args.img_size[0]))
        secondary = torch.from_numpy(d).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(rgb_t, secondary)

    visualize_outputs(orig_rgb, outputs, args.output)
    print(f"Saved outputs to {args.output}/")
    for key in outputs:
        print(f"  {key}")


if __name__ == '__main__':
    main()
