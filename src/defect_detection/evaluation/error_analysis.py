"""False-positive / false-negative analysis with Grad-CAM and ground-truth masks.

Beyond listing errors, this answers *why* the model fails:
  * Grad-CAM shows where the evidence for "defective" came from.
  * Pointing game: is the CAM peak inside the (dilated) ground-truth defect mask?
    A high hit rate on true positives shows the model detects the defect itself rather
    than a dataset shortcut (background, lighting, position).
  * Defect size vs detection: are misses concentrated on small defects (resolution limit)?
  * Hardest examples per class are always reported, so the analysis is informative even
    when the test set has very few errors.
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
import torch  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from defect_detection.config import Config  # noqa: E402
from defect_detection.export import load_checkpoint  # noqa: E402
from defect_detection.preprocessing import pad_to_square, preprocess  # noqa: E402
from defect_detection.training.models import gradcam_layer  # noqa: E402
from defect_detection.training.trainer import spec_from_config  # noqa: E402

log = logging.getLogger(__name__)


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module) -> None:
        self.model = model.eval()
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inp, out: torch.Tensor) -> None:
        self._acts = out
        out.register_hook(lambda g: setattr(self, "_grads", g))

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, float]:
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x).squeeze()
        logit.backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)  # type: ignore[union-attr]
        cam = torch.relu((weights * self._acts).sum(1)).squeeze(0).detach().numpy()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam, float(torch.sigmoid(logit.detach()))


def _cam_to_image(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.Resampling.BILINEAR)) / 255.0


def _mask_stats(cam_full: np.ndarray, mask_path: str, dilate_px: int) -> dict:
    mask_img = Image.open(mask_path).convert("L")
    mask_img = mask_img.resize(cam_full.shape[::-1], Image.Resampling.NEAREST)
    dil = mask_img.filter(ImageFilter.MaxFilter(2 * dilate_px + 1)) if dilate_px else mask_img
    mask, dmask = np.asarray(mask_img) > 127, np.asarray(dil) > 127
    peak = np.unravel_index(np.argmax(cam_full), cam_full.shape)
    return {
        "cam_peak_in_mask": bool(dmask[peak]),
        "cam_energy_in_mask": float(cam_full[dmask].sum() / max(cam_full.sum(), 1e-8)),
        "_mask": mask,
    }


def run_error_analysis(cfg: Config, top_k_hardest: int = 4) -> dict:
    run_dir = cfg.run_dir
    out = run_dir / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(run_dir / "eval" / "predictions.csv")
    model, _ = load_checkpoint(run_dir / "checkpoint.pt")
    cam_engine = GradCAM(model, gradcam_layer(model, cfg.train.backbone))
    spec = spec_from_config(cfg)
    threshold = json.loads((run_dir / "model" / "metadata.json").read_text())["threshold"]
    split = pd.read_csv(run_dir / "split.csv").set_index("image_id")
    preds["mask_path"] = preds["image_id"].map(split["mask_path"]).fillna("")

    records, cache = [], {}
    for _, r in preds.iterrows():
        image = pad_to_square(Image.open(cfg.data.raw_dir / r["path"]).convert("RGB"))
        cam, p_torch = cam_engine(torch.from_numpy(preprocess(image, spec)).unsqueeze(0))
        disp = image.resize((320, 320))
        cam_full = _cam_to_image(cam, disp.size)
        rec = {"image_id": r["image_id"], "p_torch": p_torch}
        mask = None
        if r["label"] == 1 and r["mask_path"]:
            ms = _mask_stats(cam_full, r["mask_path"], dilate_px=6)
            mask = ms.pop("_mask")
            rec.update(ms)
        records.append(rec)
        cache[r["image_id"]] = (disp, cam_full, mask)
    df = preds.merge(pd.DataFrame(records), on="image_id")
    df["margin"] = (df["defect_probability"] - threshold).abs()
    df.to_csv(out / "per_image_analysis.csv", index=False)

    fps = df[df.outcome == "FP"].sort_values("defect_probability", ascending=False)
    fns = df[df.outcome == "FN"].sort_values("defect_probability")
    hardest_normal = df[df.label == 0].nlargest(top_k_hardest, "defect_probability")
    hardest_defect = df[df.label == 1].nsmallest(top_k_hardest, "defect_probability")
    correct = (
        df[df.outcome == "TP"]
        .sort_values("defect_probability", ascending=False)
        .groupby("defect_type")
        .head(cfg.evaluation.gradcam_correct_per_type)
    )

    _gallery(pd.concat([fps, fns]), cache, out / "errors_gradcam.png", "Misclassified test images")
    _gallery(pd.concat([hardest_normal, hardest_defect]), cache, out / "hardest_cases_gradcam.png",
             "Hardest normals (highest P) and hardest defects (lowest P)")
    _gallery(correct, cache, out / "correct_detections_gradcam.png", "True positives - where the model looks")

    defects = df[(df.label == 1) & df.mask_area_frac.notna()]
    loc = defects.groupby("outcome").agg(
        n=("image_id", "size"),
        cam_peak_in_mask=("cam_peak_in_mask", "mean"),
        cam_energy_in_mask=("cam_energy_in_mask", "mean"),
        median_defect_area_pct=("mask_area_frac", lambda s: 100 * s.median()),
    ).round(3)
    by_type_loc = defects.groupby("defect_type").agg(
        n=("image_id", "size"),
        recall=("predicted", "mean"),
        cam_peak_in_mask=("cam_peak_in_mask", "mean"),
        median_defect_area_pct=("mask_area_frac", lambda s: 100 * s.median()),
        min_p=("defect_probability", "min"),
    ).round(3)
    summary = {
        "threshold": threshold,
        "false_positives": int(len(fps)),
        "false_negatives": int(len(fns)),
        "near_threshold_cases": int((df.margin < 0.15).sum()),
        "pointing_game_hit_rate_tp": float(defects.loc[defects.outcome == "TP", "cam_peak_in_mask"].mean())
        if (defects.outcome == "TP").any() else None,
        "max_torch_onnx_prob_diff": float((df.p_torch - df.defect_probability).abs().max()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    cols = ["image_id", "defect_type", "defect_probability", "margin", "mask_area_frac", "cam_peak_in_mask"]

    def table(d: pd.DataFrame) -> str:
        return d[cols].round(4).to_markdown(index=False) if len(d) else "_none_"

    md = [
        "# Error analysis (test set)",
        "",
        f"Threshold {threshold:.3f}. False positives: **{len(fps)}**, false negatives: **{len(fns)}**, "
        f"cases within ±0.15 of threshold: {summary['near_threshold_cases']}.",
        "",
        "## False negatives (escaped defects)",
        "",
        table(fns),
        "",
        "## False positives (false rejects)",
        "",
        table(fps),
        "",
        "## Hardest cases (even if correctly classified)",
        "",
        "Lowest-scoring defects:",
        "",
        table(hardest_defect),
        "",
        "Highest-scoring normals:",
        "",
        table(hardest_normal),
        "",
        "## Localisation sanity check (Grad-CAM vs ground-truth mask)",
        "",
        "`cam_peak_in_mask` = share of images whose Grad-CAM maximum lies inside the defect mask "
        "(dilated 6px at 320px display). High values on TPs mean the model keys on the defect, "
        "not on a shortcut.",
        "",
        loc.to_markdown(),
        "",
        by_type_loc.to_markdown(),
        "",
        "![errors](errors_gradcam.png)",
        "",
        "![hardest](hardest_cases_gradcam.png)",
        "",
        "![correct](correct_detections_gradcam.png)",
        "",
    ]
    (out / "error_analysis.md").write_text("\n".join(md), encoding="utf-8")
    log.info("error analysis done", extra=summary)
    return summary


def _gallery(rows: pd.DataFrame, cache: dict, path: Path, title: str, max_rows: int = 12) -> None:
    rows = rows.head(max_rows)
    if rows.empty:
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{title}: none", ha="center", va="center")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(5.2, 2.7 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, (_, r) in enumerate(rows.iterrows()):
        disp, cam, mask = cache[r["image_id"]]
        axes[i, 0].imshow(disp)
        axes[i, 1].imshow(disp)
        axes[i, 1].imshow(cam, cmap="jet", alpha=0.45)
        if mask is not None:
            for ax in axes[i]:
                ax.contour(mask, levels=[0.5], colors="white", linewidths=1.0)
        axes[i, 0].set_title(
            f"{r['outcome']} | {r['class_name']}/{r['defect_type']}\nP(defect)={r['defect_probability']:.3f}",
            fontsize=8, loc="left",
        )
        axes[i, 1].set_title("Grad-CAM (white = GT mask)", fontsize=8, loc="left")
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / len(rows)))
    fig.savefig(path, dpi=90)
    plt.close(fig)
