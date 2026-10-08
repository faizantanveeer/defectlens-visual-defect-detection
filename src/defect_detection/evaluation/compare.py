"""Backbone comparison: identical split, augmentation, imbalance handling and threshold
strategy; only the architecture changes. Produces the accuracy / latency / size table
used to justify the production model choice."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from defect_detection.config import load_config  # noqa: E402

log = logging.getLogger(__name__)


def _run_one(cfg) -> None:
    from defect_detection.evaluation.benchmark import run_benchmark
    from defect_detection.evaluation.error_analysis import run_error_analysis
    from defect_detection.evaluation.evaluate import run_evaluation
    from defect_detection.export import export_bundle
    from defect_detection.training.trainer import run_training

    run_training(cfg)
    export_bundle(cfg)
    run_evaluation(cfg)
    run_error_analysis(cfg)
    run_benchmark(cfg)


def compare_backbones(
    config_path: str, backbones: list[str], reports_dir: Path, extra_sets: list[str], skip_existing: bool = False
) -> pd.DataFrame:
    base_overrides = {}
    for item in extra_sets:
        k, _, v = item.partition("=")
        base_overrides[k] = yaml.safe_load(v)
    rows = []
    for bb in backbones:
        cfg = load_config(config_path, {**base_overrides, "train.backbone": bb, "experiment_name": bb})
        if not (skip_existing and (cfg.run_dir / "benchmark.json").exists()):
            log.info("comparison run", extra={"backbone": bb})
            _run_one(cfg)
        rows.append(_collect(cfg.run_dir))
    table = pd.DataFrame(rows)
    reports_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(reports_dir / "model_comparison.csv", index=False)
    (reports_dir / "model_comparison.md").write_text(
        "# Backbone comparison (same split / augmentation / threshold strategy)\n\n"
        + table.to_markdown(index=False, floatfmt=".3f")
        + "\n\nLatency: ONNX Runtime, CPU, batch 1, model inference only (preprocessing excluded).\n"
        + "\n![tradeoff](model_tradeoff.png)\n",
        encoding="utf-8",
    )
    _plot(table, reports_dir / "model_tradeoff.png")
    return table


def _collect(run_dir: Path) -> dict:
    ts = json.loads((run_dir / "train_summary.json").read_text())
    m = json.loads((run_dir / "eval" / "metrics.json").read_text())
    b = json.loads((run_dir / "benchmark.json").read_text())
    a = m["at_threshold"]
    return {
        "backbone": ts["backbone"],
        "params_M": ts["parameters"] / 1e6,
        "onnx_MB": b["onnx_size_mb"],
        "val_AP": ts["best_val_ap"],
        "test_AP": m["average_precision"],
        "test_ROC_AUC": m["roc_auc"],
        "threshold": m["threshold"],
        "precision": a["precision"],
        "recall": a["recall"],
        "f1": a["f1"],
        "FP": a["fp"],
        "FN": a["fn"],
        "onnx_p50_ms": b["onnx_batch1_inference"]["p50_ms"],
        "onnx_p95_ms": b["onnx_batch1_inference"]["p95_ms"],
        "torch_p50_ms": (b.get("torch_eager_batch1_inference") or {}).get("p50_ms"),
        "best_epoch": ts["best_epoch"],
        "train_min": ts["train_seconds"] / 60,
    }


def _plot(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.scatter(table.onnx_p50_ms, table.test_AP, s=table.params_M * 25 + 30, color="#2a7ab0", alpha=0.75)
    for _, r in table.iterrows():
        ax.annotate(f"{r.backbone}\nF1={r.f1:.2f}", (r.onnx_p50_ms, r.test_AP), textcoords="offset points",
                    xytext=(6, -4), fontsize=8)
    ax.set_xlabel("CPU latency p50, batch 1 (ms)")
    ax.set_ylabel("test average precision")
    ax.set_title("Accuracy vs latency (bubble = parameter count)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
