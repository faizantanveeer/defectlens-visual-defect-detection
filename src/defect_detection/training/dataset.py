"""PyTorch dataset + augmentation. Augmentations operate on PIL images and are followed by
the shared :func:`defect_detection.preprocessing.preprocess`, so the eval path is
byte-for-byte the same transform the API applies."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import v2 as T

from defect_detection.config import AugmentConfig
from defect_detection.preprocessing import PreprocessSpec, preprocess


def build_augmentation(cfg: AugmentConfig) -> Callable[[Image.Image], Image.Image]:
    ops: list = []
    if cfg.hflip:
        ops.append(T.RandomHorizontalFlip())
    if cfg.vflip:
        ops.append(T.RandomVerticalFlip())
    if cfg.rotation_degrees > 0:
        ops.append(T.RandomRotation(cfg.rotation_degrees, interpolation=T.InterpolationMode.BILINEAR))
    if cfg.translate > 0 or cfg.scale != (1.0, 1.0):
        ops.append(
            T.RandomAffine(
                degrees=0,
                translate=(cfg.translate, cfg.translate),
                scale=cfg.scale,
                interpolation=T.InterpolationMode.BILINEAR,
            )
        )
    # Brightness / contrast only: hue or saturation jitter would erase colour defects.
    if cfg.brightness > 0 or cfg.contrast > 0:
        ops.append(T.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast))
    if cfg.blur_prob > 0:
        ops.append(T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=cfg.blur_prob))
    return T.Compose(ops)


class DefectDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        raw_dir: Path,
        spec: PreprocessSpec,
        augment: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.raw_dir = Path(raw_dir)
        self.spec = spec
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.frame.iloc[idx]
        with Image.open(self.raw_dir / row["path"]) as im:
            image = im.convert("RGB")
        if self.augment is not None:
            image = self.augment(image)
        x = torch.from_numpy(preprocess(image, self.spec))
        return x, torch.tensor(float(row["label"])), idx


def make_weighted_sampler(labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    """Oversample the minority class so each batch is ~50/50 in expectation."""
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = (1.0 / counts)[labels]
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(labels), replacement=True, generator=gen
    )
