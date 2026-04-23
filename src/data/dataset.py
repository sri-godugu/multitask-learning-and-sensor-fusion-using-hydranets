import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiTaskDataset(Dataset):
    """
    Generic multi-task dataset for HydraNet.

    Expected directory layout:
        root/
          images/       *.jpg or *.png
          depth/        *.png  (16-bit, mm units → divided by 1000 → metres)
          seg_labels/   *.png  (uint8 per-pixel class IDs)
          det_labels/   *.txt  (YOLO format: class cx cy w h per line, normalised)
          train.txt / val.txt  (optional: stem names, one per line)
    """

    def __init__(self, root, split='train', tasks=None, transforms=None,
                 img_size=(512, 512)):
        self.root      = root
        self.tasks     = tasks or ['detection', 'segmentation', 'depth']
        self.transforms = transforms
        self.img_size  = img_size

        split_file = os.path.join(root, f'{split}.txt')
        if os.path.exists(split_file):
            with open(split_file) as f:
                self.samples = [l.strip() for l in f if l.strip()]
        else:
            img_dir = os.path.join(root, 'images')
            self.samples = sorted(
                os.path.splitext(fn)[0]
                for fn in os.listdir(img_dir)
                if fn.lower().endswith(('.jpg', '.jpeg', '.png'))
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        img_path = self._find_image(name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        sample = {'image': img, 'name': name}

        if 'depth' in self.tasks:
            dp = os.path.join(self.root, 'depth', name + '.png')
            if os.path.exists(dp):
                sample['depth'] = cv2.imread(dp, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        if 'segmentation' in self.tasks:
            sp = os.path.join(self.root, 'seg_labels', name + '.png')
            if os.path.exists(sp):
                sample['seg'] = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)

        if 'detection' in self.tasks:
            tp = os.path.join(self.root, 'det_labels', name + '.txt')
            if os.path.exists(tp):
                boxes = []
                with open(tp) as f:
                    for line in f:
                        boxes.append(list(map(float, line.split())))
                sample['boxes'] = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), np.float32)

        if self.transforms:
            sample = self.transforms(sample)
        else:
            sample = self._default_transform(sample)

        return sample

    # ------------------------------------------------------------------
    def _find_image(self, name):
        for ext in ('.jpg', '.jpeg', '.png'):
            p = os.path.join(self.root, 'images', name + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Image not found for sample: {name}")

    def _default_transform(self, sample):
        H, W = self.img_size
        img = cv2.resize(sample['image'], (W, H))
        t   = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        sample['image'] = (t - mean) / std

        if 'depth' in sample and not isinstance(sample['depth'], torch.Tensor):
            d = cv2.resize(sample['depth'], (W, H), interpolation=cv2.INTER_NEAREST)
            sample['depth'] = torch.from_numpy(d).unsqueeze(0)

        if 'seg' in sample and not isinstance(sample['seg'], torch.Tensor):
            s = cv2.resize(sample['seg'], (W, H), interpolation=cv2.INTER_NEAREST)
            sample['seg'] = torch.from_numpy(s).long()

        return sample
