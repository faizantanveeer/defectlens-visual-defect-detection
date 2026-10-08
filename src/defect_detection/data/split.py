"""Leakage-safe, stratified train/val/test split.

* Stratified on ``label/defect_type`` - with ~20 images per defect type, a purely
  label-stratified split can leave a defect type absent from the test set, and then
  per-type recall cannot be measured.
* Grouped on ``group_id`` - exact and near-duplicate images move together.
* Deterministic for a given seed; the result is written to CSV and versioned with the run.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def make_split(
    manifest: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int
) -> pd.DataFrame:
    df = manifest[manifest["is_valid"]].copy()
    df["stratum"] = df["class_name"] + "/" + df["defect_type"]
    largest = int(df["group_id"].value_counts().max())
    if largest > 0.2 * len(df):
        raise ValueError(
            f"a single duplicate group holds {largest}/{len(df)} images; a stratified split is "
            "impossible. Lower data.near_duplicate_hamming (see reports/eda)."
        )

    # Collapse to one row per duplicate group; the group inherits its majority stratum.
    groups = (
        df.groupby("group_id")["stratum"]
        .agg(lambda s: s.value_counts().index[0])
        .rename("stratum")
        .reset_index()
    )
    # Strata too small to stratify are pooled per class rather than crashing the split.
    counts = groups["stratum"].value_counts()
    min_needed = int(np.ceil(1 / min(val_fraction, test_fraction))) + 1
    rare = counts[counts < min_needed].index
    groups["strat_key"] = groups["stratum"].where(
        ~groups["stratum"].isin(rare), groups["stratum"].str.split("/").str[0] + "/_rare"
    )

    train_val, test = train_test_split(
        groups, test_size=test_fraction, stratify=groups["strat_key"], random_state=seed
    )
    rel_val = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val["strat_key"], random_state=seed
    )
    assignment = pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]
    )[["group_id", "split"]]
    out = df.merge(assignment, on="group_id", how="left", validate="many_to_one")
    assert out["split"].notna().all()
    validate_split(out)
    log.info("split created", extra={"summary": split_summary(out).to_dict()})
    return out


def validate_split(df: pd.DataFrame) -> None:
    """Hard invariants - training refuses to start if any of these fail."""
    per_group = df.groupby("group_id")["split"].nunique()
    if (per_group > 1).any():
        raise AssertionError("duplicate group spans multiple splits (leakage)")
    per_hash = df.groupby("sha256")["split"].nunique()
    if (per_hash > 1).any():
        raise AssertionError("identical file content present in multiple splits (leakage)")
    for split in ("train", "val", "test"):
        labels = set(df.loc[df["split"] == split, "label"])
        if labels != {0, 1}:
            raise AssertionError(f"split '{split}' does not contain both classes: {labels}")


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    stratum = (df["class_name"] + "/" + df["defect_type"]).rename("class/defect_type")
    table = pd.crosstab(stratum, df["split"], margins=True)
    return table.reindex(columns=["train", "val", "test", "All"])
