import json

import pytest

from defect_detection.inference.predictor import DefectPredictor, ModelLoadError
from defect_detection.preprocessing import decode_image


def test_predict_semantics(tiny_bundle, bright_png, dark_png):
    pred = DefectPredictor(tiny_bundle)
    (bright, dark), timing = pred.predict([decode_image(bright_png), decode_image(dark_png)])
    assert bright.predicted_class == "defective"
    assert bright.confidence == pytest.approx(bright.defect_probability)
    assert dark.predicted_class == "normal"
    assert dark.confidence == pytest.approx(1 - dark.defect_probability)
    assert timing.inference_ms >= 0


def test_review_flag_near_threshold(tiny_bundle):
    from tests.conftest import png_bytes

    pred = DefectPredictor(tiny_bundle)
    (p,), _ = pred.predict([decode_image(png_bytes((133, 133, 133)))])  # ~0.5
    assert p.review_recommended


def test_threshold_override(tiny_bundle, bright_png):
    pred = DefectPredictor(tiny_bundle, threshold_override=0.9)
    (p,), _ = pred.predict([decode_image(bright_png)])
    assert p.threshold == 0.9 and p.predicted_class == "normal"


def test_checksum_mismatch_refuses_to_load(tiny_bundle):
    meta = json.loads((tiny_bundle / "metadata.json").read_text())
    meta["model_sha256"] = "0" * 64
    (tiny_bundle / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ModelLoadError, match="checksum"):
        DefectPredictor(tiny_bundle)


def test_missing_bundle(tmp_path):
    with pytest.raises(ModelLoadError):
        DefectPredictor(tmp_path)


def test_empty_batch(tiny_bundle):
    preds, _ = DefectPredictor(tiny_bundle).predict([])
    assert preds == []
