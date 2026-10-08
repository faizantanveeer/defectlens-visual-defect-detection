"""Fast checks on training components (no real training)."""

import pytest

torch = pytest.importorskip("torch")

from defect_detection.config import AugmentConfig, Config, load_config  # noqa: E402
from defect_detection.training.dataset import build_augmentation  # noqa: E402
from defect_detection.training.models import ProbabilityWrapper, build_model  # noqa: E402


def test_default_augmentation_has_no_mirroring_or_hue_jitter():
    """Mirroring a chiral part fabricates the 'flip' defect; hue jitter erases 'color'."""
    aug = build_augmentation(AugmentConfig())
    names = [type(t).__name__ for t in aug.transforms]
    assert "RandomHorizontalFlip" not in names and "RandomVerticalFlip" not in names
    jitter = next(t for t in aug.transforms if type(t).__name__ == "ColorJitter")
    assert jitter.hue is None and jitter.saturation is None


@pytest.mark.parametrize("backbone", ["mobilenet_v3_large", "efficientnet_b0", "resnet18"])
def test_models_output_single_probability(backbone):
    m = ProbabilityWrapper(build_model(backbone, pretrained=False)).eval()
    with torch.no_grad():
        p = m(torch.zeros(2, 3, 64, 64))
    assert p.shape == (2,) and ((p >= 0) & (p <= 1)).all()


def test_repo_config_is_valid():
    cfg = load_config("configs/train.yaml")
    assert cfg.augment.hflip is False and cfg.threshold.beta == 2.0


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        Config.model_validate({"train": {"bakcbone": "resnet18"}})


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("experiment_name: x\n")
    cfg = load_config(p, {"train.backbone": "resnet18", "train.batch_size": 4})
    assert cfg.train.backbone == "resnet18" and cfg.train.batch_size == 4
