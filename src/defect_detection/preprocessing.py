"""Framework-free image decoding and preprocessing.

This module is the single source of truth for the model input transform. Training,
evaluation and the serving container (which ships onnxruntime but no torch) all call
:func:`preprocess`, so there is no train/serve skew in resizing or normalisation.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Guard against decompression bombs (e.g. a 50 KB PNG that inflates to 10 GB).
DEFAULT_MAX_PIXELS = 40_000_000


class ImageDecodeError(ValueError):
    """Raised when bytes cannot be decoded into a usable image."""


@dataclass(frozen=True)
class PreprocessSpec:
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @classmethod
    def from_dict(cls, d: dict) -> PreprocessSpec:
        return cls(int(d["image_size"]), tuple(d["mean"]), tuple(d["std"]))

    def to_dict(self) -> dict:
        return {"image_size": self.image_size, "mean": list(self.mean), "std": list(self.std)}


def decode_image(
    data: bytes, *, min_side: int = 32, max_pixels: int = DEFAULT_MAX_PIXELS
) -> Image.Image:
    """Decode untrusted bytes into an RGB PIL image, validating integrity and size."""
    if not data:
        raise ImageDecodeError("empty payload")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            if width * height > max_pixels:
                raise ImageDecodeError(
                    f"image has {width * height} pixels, limit is {max_pixels}"
                )
            probe.verify()  # catches truncated / corrupted files cheaply
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)  # honour camera orientation tags
        image.load()
    except ImageDecodeError:
        raise
    except UnidentifiedImageError as exc:
        raise ImageDecodeError("unsupported or unrecognised image format") from exc
    except (OSError, SyntaxError, ValueError) as exc:
        # Client-facing message stays generic; the chained exception keeps details for logs.
        raise ImageDecodeError(f"corrupted or truncated image data ({type(exc).__name__})") from exc
    if min(image.size) < min_side:
        raise ImageDecodeError(f"image is {image.size[0]}x{image.size[1]}, min side is {min_side}")
    return to_rgb(image)


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("I;16", "I;16B", "I;16L", "I"):
        # 16-bit industrial cameras: rescale to 8-bit before colour conversion.
        arr = np.asarray(image, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / max(hi - lo, 1.0) * 255.0
        image = Image.fromarray(arr.astype(np.uint8), mode="L")
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("RGB")


def pad_to_square(image: Image.Image, fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Letterbox to square so non-square cameras do not distort part geometry."""
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def preprocess(image: Image.Image, spec: PreprocessSpec) -> np.ndarray:
    """RGB PIL image -> normalised float32 CHW array of shape (3, S, S)."""
    image = pad_to_square(to_rgb(image))
    image = image.resize((spec.image_size, spec.image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) * (1.0 / 255.0)
    arr = (arr - np.asarray(spec.mean, dtype=np.float32)) / np.asarray(spec.std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def preprocess_batch(images: Sequence[Image.Image], spec: PreprocessSpec) -> np.ndarray:
    return np.stack([preprocess(img, spec) for img in images], axis=0)


def denormalize(chw: np.ndarray, spec: PreprocessSpec) -> np.ndarray:
    """Inverse of :func:`preprocess` normalisation -> HWC uint8 (for visualisation)."""
    hwc = chw.transpose(1, 2, 0) * np.asarray(spec.std) + np.asarray(spec.mean)
    return (np.clip(hwc, 0, 1) * 255).astype(np.uint8)
