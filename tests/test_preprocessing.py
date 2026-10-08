import io

import numpy as np
import pytest
from PIL import Image

from defect_detection.preprocessing import (
    ImageDecodeError,
    PreprocessSpec,
    decode_image,
    pad_to_square,
    preprocess,
)
from tests.conftest import png_bytes

SPEC = PreprocessSpec(224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def test_preprocess_shape_dtype_and_contiguity(rng):
    img = Image.fromarray(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8))
    x = preprocess(img, SPEC)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]


def test_preprocess_matches_torchvision_reference(rng):
    """Serving (numpy) and a standard torchvision eval transform must agree."""
    T = pytest.importorskip("torchvision.transforms")
    img = Image.fromarray(rng.integers(0, 255, (700, 700, 3), dtype=np.uint8))
    ref = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                     T.ToTensor(), T.Normalize(SPEC.mean, SPEC.std)])(img).numpy()
    np.testing.assert_allclose(preprocess(img, SPEC), ref, atol=1e-5)


def test_pad_to_square_preserves_aspect():
    out = pad_to_square(Image.new("RGB", (200, 100), (255, 0, 0)))
    assert out.size == (200, 200)
    assert out.getpixel((100, 10)) == (0, 0, 0) and out.getpixel((100, 100)) == (255, 0, 0)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_decode_converts_modes_to_rgb(mode):
    buf = io.BytesIO()
    Image.new(mode, (64, 64)).save(buf, format="PNG")
    assert decode_image(buf.getvalue()).mode == "RGB"


def test_decode_16bit_grayscale():
    buf = io.BytesIO()
    Image.fromarray(np.full((64, 64), 40000, dtype=np.uint16)).save(buf, format="PNG")
    assert decode_image(buf.getvalue()).mode == "RGB"


@pytest.mark.parametrize(
    "payload",
    [b"", b"not an image at all", png_bytes()[:60]],  # empty, garbage, truncated
    ids=["empty", "garbage", "truncated"],
)
def test_decode_rejects_bad_payloads(payload):
    with pytest.raises(ImageDecodeError):
        decode_image(payload)


def test_decode_rejects_tiny_and_huge_images():
    with pytest.raises(ImageDecodeError, match="min side"):
        decode_image(png_bytes(size=8))
    with pytest.raises(ImageDecodeError, match="pixels"):
        decode_image(png_bytes(size=200), max_pixels=100 * 100)


def test_decode_applies_exif_orientation():
    img = Image.new("RGB", (100, 50))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 CW
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    assert decode_image(buf.getvalue()).size == (50, 100)
