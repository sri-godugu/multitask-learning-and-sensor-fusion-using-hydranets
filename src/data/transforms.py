import random
import cv2
import numpy as np
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize:
    def __init__(self, size=(512, 512)):
        self.size = size  # (H, W)

    def __call__(self, sample):
        H, W = self.size
        sample['image'] = cv2.resize(sample['image'], (W, H))
        if 'depth' in sample:
            sample['depth'] = cv2.resize(sample['depth'], (W, H), cv2.INTER_NEAREST)
        if 'seg' in sample:
            sample['seg'] = cv2.resize(sample['seg'], (W, H), cv2.INTER_NEAREST)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['image'] = np.fliplr(sample['image']).copy()
            if 'depth' in sample:
                sample['depth'] = np.fliplr(sample['depth']).copy()
            if 'seg' in sample:
                sample['seg'] = np.fliplr(sample['seg']).copy()
            if 'boxes' in sample and len(sample['boxes']):
                b = sample['boxes'].copy()
                b[:, 1] = 1.0 - b[:, 1]   # flip normalised cx
                sample['boxes'] = b
        return sample


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation

    def __call__(self, sample):
        img = sample['image'].astype(np.float32)
        img *= (1.0 + random.uniform(-self.brightness, self.brightness))
        sample['image'] = np.clip(img, 0, 255).astype(np.uint8)
        return sample


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std  = std

    def __call__(self, sample):
        img = sample['image'].astype(np.float32) / 255.0
        img = (img - np.array(self.mean)) / np.array(self.std)
        sample['image'] = torch.from_numpy(img).permute(2, 0, 1).float()

        if 'depth' in sample and not isinstance(sample['depth'], torch.Tensor):
            sample['depth'] = torch.from_numpy(sample['depth']).unsqueeze(0)

        if 'seg' in sample and not isinstance(sample['seg'], torch.Tensor):
            sample['seg'] = torch.from_numpy(sample['seg'].copy()).long()

        return sample


def build_train_transforms(img_size=(512, 512)):
    return Compose([
        Resize(img_size),
        RandomHorizontalFlip(),
        ColorJitter(),
        Normalize(),
    ])


def build_val_transforms(img_size=(512, 512)):
    return Compose([Resize(img_size), Normalize()])
