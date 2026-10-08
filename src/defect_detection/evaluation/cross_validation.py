"""K-fold cross-validation with out-of-fold (OOF) predictions.

A single 67-image test split cannot distinguish a 100%-recall model from a 90% one. CV
gives every one of the N images a prediction from a model that never saw it, which:
  * tightens confidence intervals (N trials instead of ~0.2N),
  * exposes fold-to-fold variance (training stability),
  * surfaces real false positives / negatives for error analysis.

Each fold: StratifiedGroupKFold (class x defect type, duplicate groups intact) for test,
an inner stratified-group split for validation (early stopping + threshold), then the
fold's *exported ONNX bundle* scores the held-out fold.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from defect_detection.config import Config
from defect_detection.data.split import validate_split
from defect_detection.evaluation.metrics import (
    exact_rate_cis,
    ranking_metrics,
    threshold_metrics,
)
from defect_detection.inference.predictor import DefectPredictor
from defect_detection.preprocessing import pad_to_square, preprocess, preprocess_batch

log = logging.getLogger(__name__)


def make_cv_splits(df: pd.DataFrame, folds: int, inner: int, seed: int) -> list[pd.DataFrame]:
    df = df.copy()
    df["stratum"] = df["class_name"] + "/" + df["defect_type"]
    outer = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = []
    for _k, (tv_idx, te_idx) in enumerate(outer.split(df, df["stratum"], df["group_id"])):
        fold = df.copy()
        fold["split"] = "train"
        fold.iloc[te_idx, fold.columns.get_loc("split")] = "test"
        tv = df.iloc[tv_idx]
        inner_cv = StratifiedGroupKFold(n_splits=inner, shuffle=True, random_state=seed)
        _, va_rel = next(inner_cv.split(tv, tv["stratum"], tv["group_id"]))
        fold.loc[tv.index[va_rel], "split"] = "val"
        validate_split(fold)
        out.append(fold)
    return out


def run_cross_validation(cfg: Config, folds: int = 5, inner: int = 5) -> dict:
    from defect_detection.export import export_bundle
    from defect_detection.training.trainer import run_training

    base = pd.read_csv(cfg.data.split_path)
    out_dir = cfg.artifacts_dir / "cv" / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    oof, fold_rows = [], []
    for k, fold in enumerate(make_cv_splits(base, folds, inner, cfg.data.seed)):
        split_path = out_dir / f"split_fold{k}.csv"
        fold.to_csv(split_path, index=False)
        fcfg = cfg.model_copy(deep=True)
        fcfg.experiment_name = f"{cfg.experiment_name}__cv{k}"
        fcfg.data.split_path = split_path
        summary = run_training(fcfg)
        export_bundle(fcfg)
        predictor = DefectPredictor(fcfg.run_dir / "model")
        test = fold[fold.split == "test"].reset_index(drop=True)
        p = np.concatenate([
            predictor.predict_proba_array(preprocess_batch(
                [Image.open(cfg.data.raw_dir / q).convert("RGB") for q in test["path"].iloc[i : i + 16]],
                predictor.spec))
            for i in range(0, len(test), 16)
        ])
        t = predictor.threshold
        m = threshold_metrics(test["label"].to_numpy(), p, t)
        fold_rows.append({"fold": k, "n": len(test), "threshold": t, "best_epoch": summary["best_epoch"],
                          **{x: m[x] for x in ("precision", "recall", "specificity", "f1", "tp", "fp", "fn")},
                          **ranking_metrics(test["label"].to_numpy(), p)})
        oof.append(test.assign(fold=k, defect_probability=p, threshold=t, predicted=(p >= t).astype(int)))
        log.info("cv fold done", extra=fold_rows[-1])

    oof_df = pd.concat(oof, ignore_index=True)
    y, pred = oof_df["label"].to_numpy(), oof_df["predicted"].to_numpy()
    oof_df["outcome"] = np.select([(y == 1) & (pred == 1), (y == 0) & (pred == 0), (y == 0) & (pred == 1)],
                                  ["TP", "TN", "FP"], default="FN")
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    pooled = threshold_metrics(y, pred.astype(float), 0.5)  # predictions already thresholded per fold
    folds_df = pd.DataFrame(fold_rows)
    per_type = oof_df[oof_df.label == 1].groupby("defect_type").agg(
        n=("label", "size"), detected=("predicted", "sum"), min_p=("defect_probability", "min"))
    per_type["recall"] = per_type.detected / per_type.n
    result = {
        "folds": folds,
        "n_images": int(len(oof_df)),
        "pooled_oof_at_fold_thresholds": pooled,
        "pooled_ci95_exact": exact_rate_cis(pooled),
        # Pooled ranking metrics mix differently calibrated fold models: indicative only.
        "pooled_ranking_indicative": ranking_metrics(y, oof_df["defect_probability"].to_numpy()),
        "per_fold": fold_rows,
        "fold_mean": folds_df[["precision", "recall", "specificity", "f1", "average_precision"]].mean().to_dict(),
        "fold_std": folds_df[["precision", "recall", "specificity", "f1", "average_precision"]].std().to_dict(),
        "per_defect_type": per_type.reset_index().to_dict(orient="records"),
    }
    (out_dir / "cv_summary.json").write_text(json.dumps(result, indent=2, default=float))

    errors = oof_df[oof_df.outcome.isin(["FP", "FN"])].copy()
    hardest = pd.concat([
        oof_df[oof_df.label == 1].nsmallest(4, "defect_probability"),
        oof_df[oof_df.label == 0].nlargest(4, "defect_probability"),
    ])
    _cam_gallery(cfg, errors, out_dir / "oof_errors_gradcam.png", "Cross-validation errors (out-of-fold)")
    _cam_gallery(cfg, hardest, out_dir / "oof_hardest_gradcam.png", "Hardest out-of-fold cases")
    _write_report(result, folds_df, per_type, errors, out_dir / "cv_report.md")
    return result


def _cam_gallery(cfg: Config, rows: pd.DataFrame, path, title: str) -> None:
    from defect_detection.evaluation.error_analysis import GradCAM, _cam_to_image, _gallery
    from defect_detection.export import load_checkpoint
    from defect_detection.training.models import gradcam_layer
    from defect_detection.training.trainer import spec_from_config

    spec = spec_from_config(cfg)
    cache, engines = {}, {}
    rows = rows.copy()
    for _, r in rows.iterrows():
        k = int(r["fold"])
        if k not in engines:
            model, _ = load_checkpoint(cfg.artifacts_dir / "runs" / f"{cfg.experiment_name}__cv{k}" / "checkpoint.pt")
            engines[k] = GradCAM(model, gradcam_layer(model, cfg.train.backbone))
        image = pad_to_square(Image.open(cfg.data.raw_dir / r["path"]).convert("RGB"))
        cam, _ = engines[k](torch.from_numpy(preprocess(image, spec)).unsqueeze(0))
        disp = image.resize((320, 320))
        mask = None
        if isinstance(r.get("mask_path"), str) and r["mask_path"]:
            mask = np.asarray(Image.open(r["mask_path"]).convert("L").resize(disp.size)) > 127
        cache[r["image_id"]] = (disp, _cam_to_image(cam, disp.size), mask)
    _gallery(rows, cache, path, title)


def _write_report(result: dict, folds_df: pd.DataFrame, per_type: pd.DataFrame, errors: pd.DataFrame, path) -> None:
    p, ci = result["pooled_oof_at_fold_thresholds"], result["pooled_ci95_exact"]
    cols = ["image_id", "fold", "class_name", "defect_type", "defect_probability", "threshold", "outcome"]
    lines = [
        f"# {result['folds']}-fold cross-validation (out-of-fold, {result['n_images']} images)",
        "",
        "Every image is scored by a model that never saw it (its fold's exported ONNX bundle, "
        "threshold tuned on that fold's validation split).",
        "",
        "| pooled OOF metric | value | 95% exact CI |",
        "|---|---|---|",
        f"| Recall (defect catch rate) | {p['recall']:.3f} ({p['tp']}/{p['tp'] + p['fn']}) "
        f"| [{ci['recall'][0]:.3f}, {ci['recall'][1]:.3f}] |",
        f"| Specificity (good pass rate) | {p['specificity']:.3f} ({p['tn']}/{p['tn'] + p['fp']}) "
        f"| [{ci['specificity'][0]:.3f}, {ci['specificity'][1]:.3f}] |",
        f"| Precision | {p['precision']:.3f} | [{ci['precision'][0]:.3f}, {ci['precision'][1]:.3f}] |",
        f"| F1 | {p['f1']:.3f} | |",
        f"| ROC-AUC / AP (pooled, indicative) | {result['pooled_ranking_indicative']['roc_auc']:.3f} / "
        f"{result['pooled_ranking_indicative']['average_precision']:.3f} | |",
        "",
        "## Per fold",
        "",
        folds_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Per defect type (OOF)",
        "",
        per_type.to_markdown(floatfmt=".3f"),
        "",
        "## Out-of-fold errors",
        "",
        errors[cols].to_markdown(index=False, floatfmt=".4f") if len(errors) else "_none_",
        "",
        "![errors](oof_errors_gradcam.png)",
        "",
        "![hardest](oof_hardest_gradcam.png)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
