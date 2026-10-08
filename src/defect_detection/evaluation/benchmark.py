"""CPU latency / throughput benchmark of the deployable bundle (ONNX Runtime) vs PyTorch eager.

Preprocessing is measured separately: for small models it is often a large share of
end-to-end latency, which changes where optimisation effort should go.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from defect_detection.config import Config
from defect_detection.inference.predictor import DefectPredictor
from defect_detection.preprocessing import preprocess

log = logging.getLogger(__name__)


def _percentiles(samples_ms: list[float]) -> dict:
    a = np.asarray(samples_ms)
    return {"p50_ms": round(float(np.percentile(a, 50)), 3), "p95_ms": round(float(np.percentile(a, 95)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3), "mean_ms": round(float(a.mean()), 3)}


def run_benchmark(cfg: Config, iters: int = 200, threads: int | None = None) -> dict:
    run_dir = cfg.run_dir
    bundle = run_dir / "model"
    predictor = DefectPredictor(bundle, intra_op_threads=threads)
    split = pd.read_csv(run_dir / "split.csv")
    image = Image.open(cfg.data.raw_dir / split.iloc[0]["path"]).convert("RGB")
    s = predictor.spec.image_size

    pre = []
    for _ in range(min(iters, 100)):
        t = time.perf_counter()
        x = preprocess(image, predictor.spec)
        pre.append((time.perf_counter() - t) * 1000)

    predictor.warmup(10)
    one = x[None]
    lat = []
    for _ in range(iters):
        t = time.perf_counter()
        predictor.predict_proba_array(one)
        lat.append((time.perf_counter() - t) * 1000)

    throughput = {}
    for bs in (8, 32):
        batch = np.repeat(one, bs, axis=0)
        predictor.predict_proba_array(batch)
        n_rep = max(3, iters // bs)
        t = time.perf_counter()
        for _ in range(n_rep):
            predictor.predict_proba_array(batch)
        throughput[f"batch_{bs}_images_per_s"] = round(bs * n_rep / (time.perf_counter() - t), 1)

    torch_lat = None
    try:
        import torch

        from defect_detection.export import load_checkpoint

        model, _ = load_checkpoint(run_dir / "checkpoint.pt")
        xt = torch.from_numpy(one)
        with torch.inference_mode():
            for _ in range(5):
                model(xt)
            samples = []
            for _ in range(min(iters, 100)):
                t = time.perf_counter()
                model(xt)
                samples.append((time.perf_counter() - t) * 1000)
        torch_lat = _percentiles(samples)
    except Exception as exc:  # noqa: BLE001 - torch is optional in serving envs
        log.warning("torch benchmark skipped", extra={"reason": str(exc)})

    result = {
        "model_version": predictor.version,
        "input_size": s,
        "cpu_count": os.cpu_count(),
        "ort_intra_op_threads": threads or "default",
        "onnx_size_mb": round((bundle / "model.onnx").stat().st_size / 1e6, 2),
        "preprocess_700px_to_input": _percentiles(pre),
        "onnx_batch1_inference": _percentiles(lat),
        "torch_eager_batch1_inference": torch_lat,
        **throughput,
    }
    Path(run_dir / "benchmark.json").write_text(json.dumps(result, indent=2))
    log.info("benchmark", extra={"onnx_p50": result["onnx_batch1_inference"]["p50_ms"]})
    return result
