"""Export a trained checkpoint to a self-describing ONNX model bundle and verify it.

Parity is checked numerically against PyTorch on real validation images; export fails
loudly if the ONNX graph diverges, rather than shipping a silently different model.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from defect_detection import CLASS_NAMES, __version__
from defect_detection.config import Config
from defect_detection.inference.predictor import DefectPredictor, sha256_file
from defect_detection.preprocessing import preprocess_batch
from defect_detection.training.models import ProbabilityWrapper, build_model
from defect_detection.training.trainer import spec_from_config

log = logging.getLogger(__name__)

OPSET = 17
PARITY_ATOL = 1e-4


def load_checkpoint(path: Path) -> tuple[torch.nn.Module, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(state["backbone"], pretrained=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, state


def export_bundle(cfg: Config, out_dir: Path | None = None) -> Path:
    run_dir = cfg.run_dir
    out_dir = out_dir or run_dir / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    model, state = load_checkpoint(run_dir / "checkpoint.pt")
    summary = json.loads((run_dir / "train_summary.json").read_text())
    spec = spec_from_config(cfg)

    wrapper = ProbabilityWrapper(model).eval()
    dummy = torch.zeros(1, 3, spec.image_size, spec.image_size)
    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["image"],
        output_names=["defect_probability"],
        dynamic_axes={"image": {0: "batch"}, "defect_probability": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )

    created = datetime.now(timezone.utc)
    metadata = {
        "model_version": f"{cfg.train.backbone}-{created:%Y%m%d%H%M%S}",
        "package_version": __version__,
        "created_utc": created.isoformat(),
        "backbone": cfg.train.backbone,
        "class_names": list(CLASS_NAMES),
        "positive_class": CLASS_NAMES[1],
        "preprocess": spec.to_dict(),
        "input": {"name": "image", "layout": "NCHW", "dtype": "float32"},
        "output": {"name": "defect_probability", "meaning": "P(defective)"},
        "threshold": summary["threshold"],
        "threshold_strategy": summary["threshold_strategy"],
        "review_margin": cfg.threshold.review_margin,
        "best_epoch": summary["best_epoch"],
        "val_average_precision": summary["best_val_ap"],
        "experiment_name": cfg.experiment_name,
        "onnx_opset": OPSET,
        "model_sha256": sha256_file(onnx_path),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    max_diff = verify_parity(model, out_dir, cfg)
    metadata["parity_max_abs_diff"] = max_diff
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info(
        "exported",
        extra={"path": str(onnx_path), "mb": round(onnx_path.stat().st_size / 1e6, 2), "parity": max_diff},
    )
    return out_dir


def verify_parity(model: torch.nn.Module, bundle_dir: Path, cfg: Config, n: int = 16) -> float:
    split = pd.read_csv(cfg.run_dir / "split.csv")
    rows = split[split.split == "val"].head(n)
    images = [Image.open(cfg.data.raw_dir / p).convert("RGB") for p in rows["path"]]
    spec = spec_from_config(cfg)
    batch = preprocess_batch(images, spec)
    with torch.no_grad():
        ref = torch.sigmoid(model(torch.from_numpy(batch))).squeeze(1).numpy()
    predictor = DefectPredictor(bundle_dir)
    got = predictor.predict_proba_array(batch)
    single = np.array([predictor.predict_proba_array(batch[i : i + 1])[0] for i in range(len(batch))])
    max_diff = float(max(np.abs(ref - got).max(), np.abs(ref - single).max()))
    if max_diff > PARITY_ATOL:
        raise RuntimeError(f"ONNX/PyTorch parity check failed: max |diff| = {max_diff:.2e}")
    return max_diff
