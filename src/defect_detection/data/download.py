"""Fetch the public stand-in dataset (MVTec AD ``metal_nut``) into the canonical layout.

The rest of the pipeline is dataset-agnostic: any client dataset dropped into

    data/raw/normal/<images>
    data/raw/defective/[<defect_type>/]<images>
    data/masks/<defect_type>/<stem>_mask.png      (optional, enables localisation analysis)

is handled identically. This module only exists to make the public benchmark reproducible.

MVTec AD is licensed CC BY-NC-SA 4.0 (Bergmann et al., CVPR 2019).
"""

from __future__ import annotations

import io
import logging
import urllib.request
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

HF_REPO = "TheoM55/mvtec_all_objects_split"
HF_URL = "https://huggingface.co/datasets/{repo}/resolve/main/data/{category}.{split}-00000-of-00001.parquet"


def _fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("cached", extra={"file": str(dest)})
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading", extra={"url": url})
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)
    return dest


def download_mvtec(category: str, data_dir: Path) -> dict[str, int]:
    """Download one MVTec AD category and materialise it as image files.

    MVTec's own train/test split is intentionally discarded: its train split holds only
    good parts (it was designed for unsupervised anomaly detection). We pool everything
    and re-split stratified by defect type in :mod:`defect_detection.data.split`.
    """
    import pyarrow.parquet as pq

    cache = data_dir / "_download"
    raw, masks = data_dir / "raw", data_dir / "masks"
    counts: dict[str, int] = {}
    for split in ("train", "test"):
        parquet = _fetch(
            HF_URL.format(repo=HF_REPO, category=category, split=split),
            cache / f"{category}.{split}.parquet",
        )
        table = pq.read_table(parquet)
        for row in table.to_pylist():
            defect = row["defect"]
            stem = f"{split}_{defect}_{Path(row['image_path']['path']).stem}"
            folder = raw / "normal" if defect == "good" else raw / "defective" / defect
            out = folder / f"{stem}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                out.write_bytes(row["image_path"]["bytes"])
            mask = row.get("mask_path")
            if defect != "good" and mask and mask.get("bytes"):
                mask_out = masks / defect / f"{stem}_mask.png"
                mask_out.parent.mkdir(parents=True, exist_ok=True)
                if not mask_out.exists():
                    Image.open(io.BytesIO(mask["bytes"])).convert("L").save(mask_out)
            counts[defect] = counts.get(defect, 0) + 1
    log.info("dataset materialised", extra={"category": category, "counts": counts})
    return counts
