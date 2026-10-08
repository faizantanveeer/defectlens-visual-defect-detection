import numpy as np
import pandas as pd
import pytest
from PIL import Image

from defect_detection.data.manifest import build_manifest
from defect_detection.data.split import make_split, validate_split


def _textured(seed: int, size: int = 96) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (12, 12, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((size, size), Image.Resampling.BILINEAR)


@pytest.fixture
def raw_dir(tmp_path):
    raw = tmp_path / "raw"
    for i in range(40):
        p = raw / "normal" / f"n{i}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        _textured(i).save(p)
    for t, dt in enumerate(["scratch", "bent"]):
        for i in range(12):
            p = raw / "defective" / dt / f"{dt}{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            _textured(1000 + 100 * t + i).save(p)
    # Leakage traps: an exact copy and a resized near-copy of n0.
    (raw / "normal" / "n0_copy.png").write_bytes((raw / "normal" / "n0.png").read_bytes())
    _textured(0).resize((80, 80)).save(raw / "normal" / "n0_resized.png")
    (raw / "defective" / "scratch" / "corrupt.png").write_bytes(b"\x89PNG broken")
    return raw


def test_manifest_labels_types_and_integrity(raw_dir):
    m = build_manifest(raw_dir)
    assert len(m) == 40 + 24 + 3
    assert m.loc[m.path.str.startswith("normal/"), "label"].eq(0).all()
    assert set(m.loc[m.label == 1, "defect_type"]) == {"scratch", "bent"}
    bad = m[~m.is_valid]
    assert list(bad.path) == ["defective/scratch/corrupt.png"]


def test_duplicates_share_a_group(raw_dir):
    m = build_manifest(raw_dir).set_index("path")
    g = m.loc["normal/n0.png", "group_id"]
    assert m.loc["normal/n0_copy.png", "group_id"] == g
    assert m.loc["normal/n0_resized.png", "group_id"] == g
    assert m.loc["normal/n1.png", "group_id"] != g


def test_split_is_stratified_leak_free_and_deterministic(raw_dir):
    m = build_manifest(raw_dir)
    a = make_split(m, 0.2, 0.2, seed=1)
    b = make_split(m, 0.2, 0.2, seed=1)
    pd.testing.assert_series_equal(a.sort_values("path").split.reset_index(drop=True),
                                   b.sort_values("path").split.reset_index(drop=True))
    assert a.is_valid.all()
    validate_split(a)
    dup_splits = a[a.path.str.startswith("normal/n0")].split.unique()
    assert len(dup_splits) == 1
    for split in ("train", "val", "test"):
        assert {"scratch", "bent"} <= set(a.loc[a.split == split, "defect_type"])


def test_validate_split_detects_leakage():
    df = pd.DataFrame({"group_id": [0, 0, 1, 2, 3, 4, 5, 6], "sha256": list("abcdefgh"),
                       "split": ["train", "test", "train", "train", "val", "val", "test", "test"],
                       "label": [0, 0, 1, 0, 1, 0, 1, 0]})
    with pytest.raises(AssertionError, match="leakage"):
        validate_split(df)


def test_split_refuses_degenerate_duplicate_group(raw_dir):
    m = build_manifest(raw_dir)
    m["group_id"] = 0
    with pytest.raises(ValueError, match="duplicate group"):
        make_split(m, 0.2, 0.2, seed=0)


def test_cv_folds_partition_data_without_leakage(raw_dir):
    pytest.importorskip("torch")
    from defect_detection.evaluation.cross_validation import make_cv_splits

    m = build_manifest(raw_dir)
    m = m[m.is_valid]
    folds = make_cv_splits(m, folds=3, inner=3, seed=0)
    test_sets = [set(f.loc[f.split == "test", "path"]) for f in folds]
    assert set().union(*test_sets) == set(m.path)          # every image tested exactly once
    assert sum(len(s) for s in test_sets) == len(m)
    for f in folds:
        validate_split(f)                                    # groups intact, both classes everywhere
