"""Exploratory data analysis -> ``reports/eda/`` (markdown + figures + JSON).

Answers the questions that drive modelling decisions: how imbalanced is the data, are
image sizes / colour statistics consistent, are there corrupt files or duplicates, how
large are defects relative to the image (sets a floor on input resolution), and do the
classes differ in trivially exploitable ways (brightness, sharpness) that would let a
model learn a shortcut instead of the defect.
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

log = logging.getLogger(__name__)


def run_eda(manifest: pd.DataFrame, raw_dir: Path, out_dir: Path, image_size: int = 224) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = manifest
    valid = df[df["is_valid"]]

    by_type = (
        df.groupby(["class_name", "defect_type"]).size().rename("count").reset_index()
    )
    n_def, n_norm = int(valid.label.sum()), int((valid.label == 0).sum())
    sizes = valid.groupby(["width", "height"]).size().rename("count").reset_index()
    stats_by_class = valid.groupby("class_name")[
        ["mean_intensity", "std_intensity", "sharpness", "mean_r", "mean_g", "mean_b"]
    ].agg(["mean", "std"]).round(2)
    mask_stats = valid.loc[valid.mask_area_frac.notna()].groupby("defect_type")[
        "mask_area_frac"
    ].describe()[["count", "min", "50%", "max"]]
    # Smallest defect expressed in pixels at model input resolution.
    min_defect_px = (
        float(np.sqrt(valid.mask_area_frac.min()) * image_size)
        if valid.mask_area_frac.notna().any()
        else None
    )
    group_sizes = valid.groupby("group_id").size()

    summary = {
        "total_files": len(df),
        "valid_images": len(valid),
        "invalid_images": int((~df.is_valid).sum()),
        "normal": n_norm,
        "defective": n_def,
        "defective_fraction": round(n_def / max(len(valid), 1), 4),
        "imbalance_ratio_normal_to_defective": round(n_norm / max(n_def, 1), 3),
        "per_type": {f"{r.class_name}/{r.defect_type}": int(r['count']) for _, r in by_type.iterrows()},
        "image_sizes": {f"{r.width}x{r.height}": int(r['count']) for _, r in sizes.iterrows()},
        "modes": valid["mode"].value_counts().to_dict(),
        "exact_duplicates": int(df.is_exact_duplicate.sum()),
        "near_duplicate_groups_gt1": int((group_sizes > 1).sum()),
        "largest_duplicate_group": int(group_sizes.max()),
        "smallest_defect_side_px_at_input": None if min_defect_px is None else round(min_defect_px, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot_class_distribution(by_type, out_dir / "class_distribution.png")
    _plot_feature_hists(valid, out_dir / "intensity_sharpness_by_class.png")
    _plot_sample_grid(valid, raw_dir, out_dir / "samples_per_type.png")
    if valid.mask_area_frac.notna().any():
        _plot_mask_area(valid, out_dir / "defect_area_by_type.png")

    md = [
        "# Dataset analysis",
        "",
        f"- Valid images: **{len(valid)}** (invalid / corrupt: {summary['invalid_images']})",
        f"- Normal: **{n_norm}**, Defective: **{n_def}** "
        f"({summary['defective_fraction']:.1%} defective, ratio {summary['imbalance_ratio_normal_to_defective']}:1)",
        f"- Image sizes: {summary['image_sizes']}; colour modes: {summary['modes']}",
        f"- Exact duplicates: {summary['exact_duplicates']}; near-duplicate groups (>1 image): "
        f"{summary['near_duplicate_groups_gt1']} (largest {summary['largest_duplicate_group']})",
        f"- Smallest annotated defect ~ **{summary['smallest_defect_side_px_at_input']} px** "
        f"square at {image_size}px model input",
        "",
        "## Counts per class / defect type",
        "",
        by_type.to_markdown(index=False),
        "",
        "## Per-class image statistics (shortcut check)",
        "",
        "If these differ strongly between classes, a model can 'cheat' on global brightness or",
        "focus instead of learning the defect.",
        "",
        stats_by_class.to_markdown(),
        "",
    ]
    if len(mask_stats):
        md += [
            "## Defect area (fraction of image covered by ground-truth mask)",
            "",
            mask_stats.round(4).to_markdown(),
            "",
        ]
    md += [
        "![class distribution](class_distribution.png)",
        "",
        "![samples](samples_per_type.png)",
        "",
        "![intensity](intensity_sharpness_by_class.png)",
        "",
    ]
    if valid.mask_area_frac.notna().any():
        md += ["![defect area](defect_area_by_type.png)", ""]
    (out_dir / "eda.md").write_text("\n".join(md), encoding="utf-8")
    log.info("EDA written", extra={"dir": str(out_dir)})
    return summary


def _plot_class_distribution(by_type: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = by_type["class_name"] + "/" + by_type["defect_type"]
    colors = ["#2a7ab0" if c == "normal" else "#c8553d" for c in by_type["class_name"]]
    ax.bar(labels, by_type["count"], color=colors)
    for i, v in enumerate(by_type["count"]):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("images")
    ax.set_title("Images per class / defect type")
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_feature_hists(valid: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, col in zip(axes, ["mean_intensity", "std_intensity", "sharpness"], strict=True):
        for name, color in (("normal", "#2a7ab0"), ("defective", "#c8553d")):
            ax.hist(valid.loc[valid.class_name == name, col], bins=25, alpha=0.6, label=name, color=color)
        ax.set_title(col)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_sample_grid(valid: pd.DataFrame, raw_dir: Path, path: Path, per_type: int = 4) -> None:
    types = valid.groupby(["class_name", "defect_type"]).size().index.tolist()
    fig, axes = plt.subplots(len(types), per_type, figsize=(per_type * 2.2, len(types) * 2.3))
    axes = np.atleast_2d(axes)
    rng = np.random.default_rng(0)
    for r, (cls, dt) in enumerate(types):
        rows = valid[(valid.class_name == cls) & (valid.defect_type == dt)]
        picks = rows.iloc[rng.choice(len(rows), size=min(per_type, len(rows)), replace=False)]
        for c in range(per_type):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(picks):
                ax.imshow(Image.open(raw_dir / picks.iloc[c]["path"]).convert("RGB").resize((256, 256)))
        axes[r, 0].set_title(f"{cls}/{dt}", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _plot_mask_area(valid: pd.DataFrame, path: Path) -> None:
    d = valid[valid.mask_area_frac.notna()]
    types = sorted(d.defect_type.unique())
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.boxplot([d.loc[d.defect_type == t, "mask_area_frac"] * 100 for t in types], tick_labels=types)
    ax.set_ylabel("% of image area")
    ax.set_title("Defect size by type (ground-truth masks)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
