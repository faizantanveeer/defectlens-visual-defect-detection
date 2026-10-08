"""Typed experiment configuration loaded from YAML.

Every training run snapshots its resolved config next to its artifacts so any
model in ``artifacts/`` can be traced back to the exact settings that produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    raw_dir: Path = Path("data/raw")
    manifest_path: Path = Path("data/manifest.csv")
    split_path: Path = Path("data/splits/split.csv")
    val_fraction: float = Field(0.15, gt=0, lt=0.5)
    test_fraction: float = Field(0.15, gt=0, lt=0.5)
    # Images whose 256-bit perceptual hashes differ by <= this many bits are treated as one
    # group and never straddle splits (prevents near-duplicate leakage).
    near_duplicate_hamming: int = Field(10, ge=0, le=256)
    seed: int = 42


class PreprocessConfig(_Strict):
    image_size: int = Field(224, ge=32, le=1024)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class AugmentConfig(_Strict):
    # Parts arrive at arbitrary rotation on the line, so full rotation is label-preserving.
    rotation_degrees: float = Field(180.0, ge=0, le=180)
    translate: float = Field(0.05, ge=0, le=0.3)
    scale: tuple[float, float] = (0.95, 1.05)
    brightness: float = Field(0.15, ge=0, le=1)
    contrast: float = Field(0.15, ge=0, le=1)
    blur_prob: float = Field(0.1, ge=0, le=1)
    # Mirroring is OFF by default: for chiral parts (e.g. the metal nut) a mirrored
    # good part is visually identical to the "flip" (upside-down) defect.
    hflip: bool = False
    vflip: bool = False


class TrainConfig(_Strict):
    backbone: Literal["mobilenet_v3_large", "efficientnet_b0", "resnet18", "resnet50"] = (
        "mobilenet_v3_large"
    )
    pretrained: bool = True
    batch_size: int = Field(16, ge=1)
    head_epochs: int = Field(3, ge=0)
    finetune_epochs: int = Field(30, ge=1)
    warmup_epochs: int = Field(2, ge=0)
    head_lr: float = Field(1e-3, gt=0)
    backbone_lr: float = Field(1e-4, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    dropout: float = Field(0.2, ge=0, lt=1)
    # Keep pretrained BatchNorm running statistics fixed while fine-tuning. With ~200
    # images and batch 16, batch statistics are noisy, and backbones with low BN momentum
    # (MobileNetV3: 0.01) otherwise train on batch stats but evaluate on stale ImageNet
    # stats -> severe train/eval mismatch. Affine BN parameters still learn.
    freeze_bn_stats: bool = True
    imbalance_strategy: Literal["pos_weight", "weighted_sampler", "none"] = "pos_weight"
    early_stopping_patience: int = Field(8, ge=1)
    num_workers: int = Field(2, ge=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class ThresholdConfig(_Strict):
    # f_beta with beta=2 weighs recall 2x precision: a missed defect that ships to a
    # customer costs more than a good part sent to manual re-inspection.
    strategy: Literal["f_beta", "min_recall", "fixed"] = "f_beta"
    beta: float = Field(2.0, gt=0)
    min_recall: float = Field(0.95, gt=0, le=1)
    fixed: float = Field(0.5, gt=0, lt=1)
    # Search range for tuned thresholds (see metrics.select_threshold for the rationale).
    search_bounds: tuple[float, float] = (0.05, 0.95)
    # Predictions with |p - threshold| < review_margin are flagged for human review.
    review_margin: float = Field(0.15, ge=0, lt=0.5)


class EvalConfig(_Strict):
    bootstrap_iterations: int = Field(2000, ge=0)
    gradcam_correct_per_type: int = Field(2, ge=0)


class Config(_Strict):
    experiment_name: str = "default"
    artifacts_dir: Path = Path("artifacts")
    data: DataConfig = DataConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    augment: AugmentConfig = AugmentConfig()
    train: TrainConfig = TrainConfig()
    threshold: ThresholdConfig = ThresholdConfig()
    evaluation: EvalConfig = EvalConfig()

    @model_validator(mode="after")
    def _check_fractions(self) -> Config:
        if self.data.val_fraction + self.data.test_fraction >= 0.6:
            raise ValueError("val_fraction + test_fraction must leave >= 40% for training")
        return self

    @property
    def run_dir(self) -> Path:
        return self.artifacts_dir / "runs" / self.experiment_name


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    import yaml

    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return Config.model_validate(raw)


def dump_config(cfg: Config, path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.model_dump(mode="json"), fh, sort_keys=False)
