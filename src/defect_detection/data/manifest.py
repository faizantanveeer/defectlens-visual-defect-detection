"""Build a validated manifest (one row per image) from the canonical folder layout.

The manifest is the contract between data and training: it records label, defect type,
integrity status, content hash and perceptual hash for every file, so splits are
reproducible and corrupt / duplicate images are visible instead of silently ingested.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
NORMAL_DIR, DEFECTIVE_DIR = "normal", "defective"


HASH_SIZE = 16  # 256-bit hash


def dhash(image: Image.Image, hash_size: int = HASH_SIZE) -> str:
    """Difference hash (hex), robust to rescaling / recompression; used for near-duplicates.

    256 bits rather than the common 64: on homogeneous product imagery (same part, same
    fixture, same lighting) 64-bit hashes of *different* parts are often only 1-3 bits
    apart, which makes every image a "duplicate" of every other.
    """
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    px = np.asarray(gray, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return np.packbits(bits).tobytes().hex()


def _inspect(path: Path, raw_dir: Path, masks_dir: Path | None) -> dict:
    rel = path.relative_to(raw_dir).as_posix()
    parts = rel.split("/")
    label = 1 if parts[0] == DEFECTIVE_DIR else 0
    defect_type = "none" if label == 0 else (parts[1] if len(parts) > 2 else "unspecified")
    row: dict = {
        "image_id": rel.rsplit(".", 1)[0].replace("/", "__"),
        "path": rel,
        "label": label,
        "class_name": parts[0],
        "defect_type": defect_type,
        "file_bytes": path.stat().st_size,
        "is_valid": True,
        "error": "",
    }
    data = path.read_bytes()
    row["sha256"] = hashlib.sha256(data).hexdigest()
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            row.update(width=im.width, height=im.height, mode=im.mode)
            rgb = im.convert("RGB")
            row["dhash"] = dhash(rgb)
            arr = np.asarray(rgb.resize((128, 128)), dtype=np.float32)
            row["mean_intensity"] = float(arr.mean())
            row["std_intensity"] = float(arr.std())
            row["mean_r"], row["mean_g"], row["mean_b"] = (float(v) for v in arr.mean((0, 1)))
            row["sharpness"] = _laplacian_var(np.asarray(rgb.convert("L"), dtype=np.float32))
    except Exception as exc:  # noqa: BLE001 - any decode failure marks the file invalid
        row.update(is_valid=False, error=f"{type(exc).__name__}: {exc}")
    row["mask_path"] = ""
    row["mask_area_frac"] = np.nan
    if label == 1 and masks_dir is not None:
        mask = masks_dir / defect_type / f"{path.stem}_mask.png"
        if mask.exists():
            row["mask_path"] = mask.as_posix()
            m = np.asarray(Image.open(mask).convert("L")) > 127
            row["mask_area_frac"] = float(m.mean())
    return row


def _laplacian_var(gray: np.ndarray) -> float:
    """Variance of the Laplacian - standard focus/blur measure."""
    lap = (
        -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return float(lap.var())


def build_manifest(
    raw_dir: Path, masks_dir: Path | None = None, max_hamming: int = 4, workers: int = 8
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    for required in (NORMAL_DIR, DEFECTIVE_DIR):
        if not (raw_dir / required).is_dir():
            raise FileNotFoundError(
                f"expected {raw_dir / required}/ - see README 'Using your own dataset'"
            )
    files = sorted(
        p
        for sub in (NORMAL_DIR, DEFECTIVE_DIR)
        for p in (raw_dir / sub).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"no images found under {raw_dir}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda p: _inspect(p, raw_dir, masks_dir), files))
    df = pd.DataFrame(rows)
    df = assign_duplicate_groups(df, max_hamming)
    n_bad = int((~df.is_valid).sum())
    if n_bad:
        log.warning("invalid images excluded", extra={"count": n_bad})
    log.info(
        "manifest built",
        extra={"images": len(df), "defective": int(df.label.sum()), "invalid": n_bad},
    )
    return df


def assign_duplicate_groups(df: pd.DataFrame, max_hamming: int = 10) -> pd.DataFrame:
    """Union-find over (exact sha256) + (256-bit dHash Hamming <= max_hamming) edges.

    Every image gets a ``group_id``; the splitter keeps groups intact so the same
    physical part can never appear in both train and test.
    O(n^2) pairwise popcount in chunks - fine for ~50k images; beyond that switch to
    a BK-tree / LSH index.
    """
    df = df.copy()
    n = len(df)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for _, idx in df.groupby("sha256").groups.items():
        idx = list(idx)
        for other in idx[1:]:
            union(idx[0], other)

    valid = df["dhash"].notna().to_numpy() if "dhash" in df else np.zeros(n, bool)
    vidx = np.flatnonzero(valid)
    if len(vidx) > 1:
        hashes = np.stack(
            [np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in df.loc[valid, "dhash"]]
        )
        chunk = 512
        for start in range(0, len(hashes), chunk):
            x = hashes[start : start + chunk, None, :] ^ hashes[None, :, :]
            dist = np.unpackbits(x, axis=-1).sum(-1)
            ii, jj = np.nonzero(dist <= max_hamming)
            for i, j in zip(ii + start, jj, strict=True):
                if i < j:
                    union(int(vidx[i]), int(vidx[j]))

    roots = [find(i) for i in range(n)]
    df["group_id"] = pd.Series(roots).astype("category").cat.codes.to_numpy()
    sizes = df.groupby("group_id")["group_id"].transform("size")
    df["is_exact_duplicate"] = df.duplicated("sha256", keep="first")
    df["near_duplicate_group_size"] = sizes
    largest = int(sizes.max()) if n else 0
    if largest > max(10, 0.05 * n):
        # Single-linkage chaining: a loose threshold transitively links distinct images.
        log.warning(
            "suspiciously large near-duplicate group - threshold likely too loose",
            extra={"largest_group": largest, "images": n, "max_hamming": max_hamming},
        )
    return df
