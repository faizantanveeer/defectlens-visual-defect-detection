"""Metrics, threshold selection and bootstrap confidence intervals.

Positive class = defective. In inspection terms:
  * recall (sensitivity)  = share of defective parts caught      -> FN = escaped defect
  * precision             = share of rejects that were truly bad -> FP = false reject
  * specificity           = share of good parts passed
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    return {
        "tp": int(((y == 1) & (pred == 1)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "tn": int(((y == 0) & (pred == 0)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()),
    }


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float, beta: float = 2.0) -> dict:
    c = confusion(y, (np.asarray(p) >= threshold).astype(int))
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    b2 = beta * beta
    return {
        "threshold": float(threshold),
        **c,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        f"f{beta:g}": _safe_div((1 + b2) * precision * recall, b2 * precision + recall),
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2,
        "false_positive_rate": _safe_div(fp, fp + tn),
        "false_negative_rate": _safe_div(fn, fn + tp),
    }


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Threshold-free metrics. AP (area under PR) is the primary model-selection metric:
    unlike accuracy or ROC-AUC it stays informative under class imbalance."""
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "average_precision": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }


@dataclass
class ThresholdChoice:
    threshold: float
    strategy: str
    val_metrics: dict


def _candidates(p: np.ndarray) -> np.ndarray:
    """Midpoints between consecutive distinct scores (+ extremes).

    Thresholding *between* observed scores rather than *at* them gives a margin on both
    sides, which matters with a validation set this small."""
    s = np.unique(np.clip(p, 0, 1))
    mids = (s[:-1] + s[1:]) / 2 if len(s) > 1 else np.array([])
    return np.unique(np.concatenate([[max(s[0] / 2, 1e-6)], mids, [min((s[-1] + 1) / 2, 1 - 1e-6)]]))


def _gap(p: np.ndarray, t: float) -> float:
    above, below = p[p >= t], p[p < t]
    return float((above.min() if len(above) else 1.0) - (below.max() if len(below) else 0.0))


def select_threshold(
    y: np.ndarray,
    p: np.ndarray,
    strategy: str = "f_beta",
    beta: float = 2.0,
    min_recall: float = 0.95,
    fixed: float = 0.5,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> ThresholdChoice:
    """Pick the decision threshold on validation scores.

    ``bounds`` restricts the search. Fine-tuned CNNs output saturated probabilities, so on a
    small validation set a single hard example at p~0.001 can make a near-zero threshold
    look free (no extra val false positives) while admitting many false rejects in
    production. 5-fold CV showed exactly this; bounding to [0.05, 0.95] removed 4 of 5
    out-of-fold false positives without losing recall.
    """
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    lo, hi = bounds
    if strategy == "fixed":
        t = fixed
    else:
        cands = _candidates(p)
        cands = np.unique(np.concatenate([cands[(cands >= lo) & (cands <= hi)], [lo, hi, 0.5]]))
        cands = cands[(cands > 0) & (cands < 1)]
        rows = [threshold_metrics(y, p, t, beta) for t in cands]
        if strategy == "f_beta":
            key = f"f{beta:g}"
            best = max(r[key] for r in rows)
            ties = [r["threshold"] for r in rows if np.isclose(r[key], best)]
        elif strategy == "min_recall":
            ok = [r for r in rows if r["recall"] >= min_recall]
            if not ok:  # unreachable target: fall back to catching everything we can
                ok = [max(rows, key=lambda r: (r["recall"], r["precision"]))]
            best = max(r["precision"] for r in ok)
            ties = [r["threshold"] for r in ok if np.isclose(r["precision"], best)]
        else:
            raise ValueError(f"unknown threshold strategy {strategy}")
        # Among equally good thresholds choose the one with the widest score gap.
        t = max(ties, key=lambda th: (_gap(p, th), -abs(th - 0.5)))
    return ThresholdChoice(float(t), strategy, threshold_metrics(y, p, t, beta))


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    n_iter: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Stratified percentile bootstrap. With ~20 defective test images a single point
    estimate hides a lot of uncertainty - the CI makes that explicit."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    if n_iter <= 0:
        return {}
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    keys = ("precision", "recall", "specificity", "f1", "average_precision", "roc_auc")
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_iter):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = {**threshold_metrics(y[idx], p[idx], threshold), **ranking_metrics(y[idx], p[idx])}
        for k in keys:
            samples[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in samples.items()}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI for a proportion k/n.

    Unlike the percentile bootstrap, this stays honest at the boundary: 19/19 caught
    defects gives [0.824, 1.0], not a degenerate [1.0, 1.0]."""
    from scipy.stats import beta

    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def exact_rate_cis(m: dict, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
    """Clopper-Pearson CIs for the count-based rates in a threshold_metrics() dict."""
    tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
    return {
        "recall": clopper_pearson(tp, tp + fn, alpha),
        "specificity": clopper_pearson(tn, tn + fp, alpha),
        "precision": clopper_pearson(tp, tp + fp, alpha),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)
