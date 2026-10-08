"""Robustness probes for the deployed bundle.

1. Perturbation sweep (inference only): how do recall / false-reject rate move when the
   line camera drifts - lighting, focus, sensor noise, compression, part rotation?
2. Unseen-defect hold-out (re-trains): a supervised classifier only knows the defect types
   it was shown. Holding one type out of training and measuring its recall estimates how
   the model behaves when a *new* failure mode appears on the line.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from defect_detection.config import Config
from defect_detection.evaluation.metrics import ranking_metrics, threshold_metrics
from defect_detection.inference.predictor import DefectPredictor
from defect_detection.preprocessing import preprocess_batch

log = logging.getLogger(__name__)


def _jpeg(q: int) -> Callable[[Image.Image], Image.Image]:
    def f(im: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return f


def _noise(sigma: float) -> Callable[[Image.Image], Image.Image]:
    def f(im: Image.Image) -> Image.Image:
        rng = np.random.default_rng(0)
        a = np.asarray(im, dtype=np.float32) + rng.normal(0, sigma, (im.height, im.width, 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return f


PERTURBATIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "clean": lambda im: im,
    "brightness_-30%": lambda im: ImageEnhance.Brightness(im).enhance(0.7),
    "brightness_+30%": lambda im: ImageEnhance.Brightness(im).enhance(1.3),
    "contrast_-30%": lambda im: ImageEnhance.Contrast(im).enhance(0.7),
    "gaussian_blur_r2": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
    "gaussian_blur_r4": lambda im: im.filter(ImageFilter.GaussianBlur(4)),
    "sensor_noise_s10": _noise(10),
    "sensor_noise_s25": _noise(25),
    "jpeg_q30": _jpeg(30),
    "rotation_45": lambda im: im.rotate(45, resample=Image.Resampling.BILINEAR),
    "downscale_x4_up": lambda im: im.resize((im.width // 4, im.height // 4)).resize(im.size),
}


def run_perturbation_sweep(cfg: Config) -> pd.DataFrame:
    run_dir = cfg.run_dir
    predictor = DefectPredictor(run_dir / "model")
    split = pd.read_csv(run_dir / "split.csv")
    test = split[split.split == "test"].reset_index(drop=True)
    y = test["label"].to_numpy()
    images = [Image.open(cfg.data.raw_dir / p).convert("RGB") for p in test["path"]]
    rows = []
    for name, fn in PERTURBATIONS.items():
        p = predictor.predict_proba_array(preprocess_batch([fn(im) for im in images], predictor.spec))
        m = threshold_metrics(y, p, predictor.threshold)
        types = sorted(test.loc[test.label == 1, "defect_type"].unique())
        per_type = {f"recall_{t}": float((p[(test.defect_type == t).to_numpy()] >= predictor.threshold).mean())
                    for t in types}
        rows.append({"perturbation": name, "recall": m["recall"], "false_reject_rate": m["false_positive_rate"],
                     "f1": m["f1"], **ranking_metrics(y, p), **per_type})
    df = pd.DataFrame(rows)
    out = run_dir / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "perturbations.csv", index=False)
    (out / "perturbations.md").write_text(
        "# Perturbation robustness (test set, fixed threshold)\n\n"
        + df.to_markdown(index=False, floatfmt=".3f") + "\n",
        encoding="utf-8",
    )
    log.info("perturbation sweep done", extra={"dir": str(out)})
    return df


def run_unseen_defect_holdout(cfg: Config, epochs: int = 15) -> pd.DataFrame:
    """For each defect type T: train without T (train+val), then measure recall on all T
    images and the false-reject rate on the normal test images."""
    from defect_detection.export import export_bundle
    from defect_detection.training.trainer import run_training

    base_split = pd.read_csv(cfg.data.split_path)
    types = sorted(base_split.loc[base_split.label == 1, "defect_type"].unique())
    rows = []
    for t in types:
        name = f"{cfg.experiment_name}__holdout_{t}"
        split = base_split.copy()
        split.loc[split.defect_type == t, "split"] = "test"  # never seen in train/val
        split_path = cfg.artifacts_dir / "holdout" / f"split_without_{t}.csv"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(split_path, index=False)
        hcfg = cfg.model_copy(deep=True)
        hcfg.experiment_name = name
        hcfg.data.split_path = split_path
        hcfg.train.finetune_epochs = epochs
        run_training(hcfg)
        export_bundle(hcfg)
        predictor = DefectPredictor(hcfg.run_dir / "model")
        test = split[split.split == "test"].reset_index(drop=True)
        p = predictor.predict_proba_array(preprocess_batch(
            [Image.open(cfg.data.raw_dir / q).convert("RGB") for q in test["path"]], predictor.spec))
        held = (test.defect_type == t).to_numpy()
        seen = ((test.label == 1) & ~held).to_numpy()
        normal = (test.label == 0).to_numpy()
        rows.append({
            "held_out_type": t,
            "n_held_out": int(held.sum()),
            "recall_unseen_type": float((p[held] >= predictor.threshold).mean()),
            "mean_p_unseen_type": float(p[held].mean()),
            "recall_seen_types": float((p[seen] >= predictor.threshold).mean()),
            "false_reject_rate": float((p[normal] >= predictor.threshold).mean()),
            "auc_unseen_vs_normal": ranking_metrics(np.r_[np.ones(held.sum()), np.zeros(normal.sum())],
                                                    np.r_[p[held], p[normal]])["roc_auc"],
            "threshold": predictor.threshold,
        })
        log.info("holdout done", extra=rows[-1])
    df = pd.DataFrame(rows)
    out = cfg.artifacts_dir / "holdout"
    df.to_csv(out / "unseen_defect_holdout.csv", index=False)
    (out / "unseen_defect_holdout.md").write_text(
        "# Unseen-defect hold-out (train without one defect type)\n\n"
        + df.to_markdown(index=False, floatfmt=".3f") + "\n", encoding="utf-8")
    (out / "unseen_defect_holdout.json").write_text(json.dumps(rows, indent=2))
    return df
