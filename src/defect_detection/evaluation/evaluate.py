"""Held-out test evaluation of the *exported* model bundle.

Evaluation deliberately runs through :class:`DefectPredictor` (ONNX Runtime + shared
preprocessing), i.e. the exact code path the API serves. Reported numbers therefore
describe the deployed artifact, not a PyTorch model that differs from it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from defect_detection.config import Config  # noqa: E402
from defect_detection.evaluation.metrics import (  # noqa: E402
    bootstrap_ci,
    exact_rate_cis,
    expected_calibration_error,
    ranking_metrics,
    threshold_metrics,
)
from defect_detection.inference.predictor import DefectPredictor  # noqa: E402
from defect_detection.preprocessing import preprocess_batch  # noqa: E402

log = logging.getLogger(__name__)


def predict_frame(predictor: DefectPredictor, frame: pd.DataFrame, raw_dir: Path, batch: int = 16) -> np.ndarray:
    probs = []
    for start in range(0, len(frame), batch):
        paths = frame["path"].iloc[start : start + batch]
        images = [Image.open(raw_dir / p).convert("RGB") for p in paths]
        probs.append(predictor.predict_proba_array(preprocess_batch(images, predictor.spec)))
    return np.concatenate(probs)


def run_evaluation(cfg: Config, split_name: str = "test") -> dict:
    run_dir = cfg.run_dir
    out = run_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    predictor = DefectPredictor(run_dir / "model")
    split = pd.read_csv(run_dir / "split.csv")
    frame = split[split.split == split_name].reset_index(drop=True)
    y = frame["label"].to_numpy().astype(int)
    p = predict_frame(predictor, frame, cfg.data.raw_dir)
    t = predictor.threshold

    frame = frame.assign(
        defect_probability=p,
        predicted=(p >= t).astype(int),
        review_recommended=np.abs(p - t) < predictor.review_margin,
    )
    frame["outcome"] = np.select(
        [(y == 1) & (frame.predicted == 1), (y == 0) & (frame.predicted == 0), (y == 0) & (frame.predicted == 1)],
        ["TP", "TN", "FP"],
        default="FN",
    )
    keep = ["image_id", "path", "label", "class_name", "defect_type", "mask_area_frac",
            "defect_probability", "predicted", "outcome", "review_recommended"]
    frame[keep].to_csv(out / "predictions.csv", index=False)

    at_t = threshold_metrics(y, p, t)
    at_05 = threshold_metrics(y, p, 0.5)
    per_type = (
        frame[frame.label == 1]
        .groupby("defect_type")
        .agg(n=("label", "size"), detected=("predicted", "sum"), mean_p=("defect_probability", "mean"))
        .assign(recall=lambda d: d.detected / d.n)
        .round(4)
    )
    metrics = {
        "split": split_name,
        "n": int(len(y)),
        "n_defective": int(y.sum()),
        "n_normal": int((y == 0).sum()),
        "model_version": predictor.version,
        "threshold": t,
        "at_threshold": at_t,
        "at_0.5": at_05,
        **ranking_metrics(y, p),
        "expected_calibration_error": expected_calibration_error(y, p),
        "review_rate": float(frame.review_recommended.mean()),
        "ci95_at_threshold": bootstrap_ci(y, p, t, cfg.evaluation.bootstrap_iterations, seed=cfg.data.seed),
        "ci95_exact_binomial": exact_rate_cis(at_t),
        "per_defect_type": per_type.reset_index().to_dict(orient="records"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _plot_confusion(at_t, out / "confusion_matrix.png", t)
    _plot_curves(y, p, t, out / "pr_roc_curves.png")
    _plot_threshold_sweep(y, p, t, out / "threshold_sweep.png")
    _plot_score_distribution(frame, t, predictor.review_margin, out / "score_distribution.png")
    _write_report(metrics, per_type, out / "report.md")
    log.info(
        "evaluation done",
        extra={k: round(at_t[k], 4) for k in ("precision", "recall", "f1", "specificity")}
        | {"ap": round(metrics["average_precision"], 4)},
    )
    return metrics


def _plot_confusion(m: dict, path: Path, t: float) -> None:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    labels = [["TN\n(good passed)", "FP\n(false reject)"], ["FN\n(escaped defect)", "TP\n(defect caught)"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{labels[i][j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks([0, 1], ["normal", "defective"])
    ax.set_yticks([0, 1], ["normal", "defective"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Confusion matrix @ t={t:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_curves(y, p, t, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.6))
    prec, rec, thr = precision_recall_curve(y, p)
    a.plot(rec, prec, color="#2a7ab0")
    m = threshold_metrics(y, p, t)
    a.scatter([m["recall"]], [m["precision"]], color="#c8553d", zorder=3, label=f"operating point t={t:.2f}")
    a.set_xlabel("recall")
    a.set_ylabel("precision")
    a.set_title("Precision-Recall")
    a.legend(loc="lower left")
    fpr, tpr, _ = roc_curve(y, p)
    b.plot(fpr, tpr, color="#2a7ab0")
    b.plot([0, 1], [0, 1], ls="--", color="#999")
    b.scatter([m["false_positive_rate"]], [m["recall"]], color="#c8553d", zorder=3)
    b.set_xlabel("false positive rate")
    b.set_ylabel("true positive rate")
    b.set_title("ROC")
    for ax in (a, b):
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_threshold_sweep(y, p, t, path: Path) -> None:
    ts = np.linspace(0.01, 0.99, 99)
    rows = [threshold_metrics(y, p, x) for x in ts]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for key, color in (("precision", "#2a7ab0"), ("recall", "#c8553d"), ("f1", "#5b8c5a")):
        ax.plot(ts, [r[key] for r in rows], label=key, color=color)
    ax.axvline(t, ls="--", color="#333", label=f"chosen t={t:.2f}")
    ax.set_xlabel("decision threshold on P(defective)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.set_title("Metric vs threshold (test)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_score_distribution(frame: pd.DataFrame, t: float, margin: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.linspace(0, 1, 41)
    ax.hist(frame.loc[frame.label == 0, "defect_probability"], bins=bins, alpha=0.7, label="normal", color="#2a7ab0")
    ax.hist(frame.loc[frame.label == 1, "defect_probability"], bins=bins, alpha=0.7, label="defective", color="#c8553d")
    ax.axvline(t, ls="--", color="#333", label="threshold")
    ax.axvspan(max(t - margin, 0), min(t + margin, 1), color="#f2c14e", alpha=0.25, label="review band")
    ax.set_xlabel("P(defective)")
    ax.set_ylabel("images")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Score distribution (test)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fmt_ci(ci: dict, k: str) -> str:
    return f"[{ci[k][0]:.3f}, {ci[k][1]:.3f}]" if k in ci else "n/a"


def _write_report(m: dict, per_type: pd.DataFrame, path: Path) -> None:
    a, ci, ex = m["at_threshold"], m["ci95_at_threshold"], m["ci95_exact_binomial"]
    lines = [
        f"# Test evaluation - `{m['model_version']}`",
        "",
        f"Test set: {m['n']} images ({m['n_defective']} defective, {m['n_normal']} normal). "
        f"Decision threshold **{m['threshold']:.3f}** (tuned on validation).",
        "",
        "| metric | value | 95% exact binomial CI | 95% bootstrap CI |",
        "|---|---|---|---|",
        f"| Precision | {a['precision']:.3f} | {_fmt_ci(ex, 'precision')} | {_fmt_ci(ci, 'precision')} |",
        f"| Recall (defect catch rate) | {a['recall']:.3f} | {_fmt_ci(ex, 'recall')} | {_fmt_ci(ci, 'recall')} |",
        f"| Specificity (good pass rate) | {a['specificity']:.3f} | {_fmt_ci(ex, 'specificity')} "
        f"| {_fmt_ci(ci, 'specificity')} |",
        f"| F1 | {a['f1']:.3f} | | {_fmt_ci(ci, 'f1')} |",
        f"| Average precision (PR-AUC) | {m['average_precision']:.3f} | | {_fmt_ci(ci, 'average_precision')} |",
        f"| ROC-AUC | {m['roc_auc']:.3f} | | {_fmt_ci(ci, 'roc_auc')} |",
        f"| Balanced accuracy | {a['balanced_accuracy']:.3f} | | |",
        f"| Expected calibration error | {m['expected_calibration_error']:.3f} | | |",
        f"| Flagged for human review | {m['review_rate']:.1%} | | |",
        "",
        "The bootstrap CI collapses to a point when there are no errors; the exact binomial CI "
        "is the honest bound for a test set this size.",
        "",
        f"Confusion @ threshold: TP={a['tp']} FP={a['fp']} TN={a['tn']} FN={a['fn']}  ",
        f"For reference @ 0.5: precision={m['at_0.5']['precision']:.3f}, recall={m['at_0.5']['recall']:.3f}, "
        f"FP={m['at_0.5']['fp']}, FN={m['at_0.5']['fn']}",
        "",
        "## Recall per defect type",
        "",
        per_type.to_markdown(),
        "",
        "![confusion](confusion_matrix.png) ![curves](pr_roc_curves.png)",
        "",
        "![sweep](threshold_sweep.png) ![scores](score_distribution.png)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
