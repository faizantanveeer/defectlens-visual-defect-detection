import numpy as np
import pytest

from defect_detection.evaluation.metrics import (
    bootstrap_ci,
    expected_calibration_error,
    ranking_metrics,
    select_threshold,
    threshold_metrics,
)


def test_threshold_metrics_counts():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.2, 0.7, 0.1, 0.1, 0.3])
    m = threshold_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(3 / 4)


def test_select_threshold_uses_max_margin_gap():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.05, 0.10, 0.20, 0.80, 0.90])
    choice = select_threshold(y, p, "f_beta", beta=2)
    assert 0.2 < choice.threshold < 0.8
    assert choice.threshold == pytest.approx(0.5)  # midpoint of the widest separating gap
    assert choice.val_metrics["recall"] == 1.0


def test_f2_prefers_recall_over_precision():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.55, 0.35, 0.6, 0.7, 0.9])
    t = select_threshold(y, p, "f_beta", beta=2).threshold
    assert threshold_metrics(y, p, t)["recall"] == 1.0  # accepts an extra FP to catch every defect


def test_min_recall_strategy():
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.3, 0.7, 0.8, 0.9])
    t = select_threshold(y, p, "min_recall", min_recall=1.0).threshold
    assert threshold_metrics(y, p, t)["recall"] == 1.0


def test_ranking_metrics_single_class_is_nan():
    assert np.isnan(ranking_metrics(np.zeros(4), np.random.rand(4))["roc_auc"])


def test_bootstrap_ci_brackets_point_estimate(rng):
    y = np.r_[np.ones(30), np.zeros(70)].astype(int)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 100), 0, 1)
    ci = bootstrap_ci(y, p, 0.5, n_iter=300)
    point = threshold_metrics(y, p, 0.5)["recall"]
    assert ci["recall"][0] <= point <= ci["recall"][1]


def test_ece_perfectly_calibrated_is_zero():
    assert expected_calibration_error(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0


def test_clopper_pearson_boundaries():
    from defect_detection.evaluation.metrics import clopper_pearson

    lo, hi = clopper_pearson(19, 19)
    assert hi == 1.0 and lo == pytest.approx(0.8235, abs=1e-3)  # zero errors != certainty
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.35


def test_threshold_bounds_ignore_single_saturated_val_example():
    # One hard val defect at p=0.001 makes a near-zero threshold "free" on val.
    y = np.array([0] * 30 + [1] * 10)
    p = np.r_[np.full(30, 1e-5), np.full(9, 0.999), [1e-3]]
    assert select_threshold(y, p, "f_beta", beta=2).threshold < 0.01
    t = select_threshold(y, p, "f_beta", beta=2, bounds=(0.05, 0.95)).threshold
    assert 0.05 <= t <= 0.95
