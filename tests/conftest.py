"""Shared fixtures. Tests never need the trained model: ``tiny_bundle`` builds a minimal
ONNX graph with the production I/O contract (P(defective) = sigmoid(mean(pixels)))."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from defect_detection.inference.predictor import sha256_file

IMAGE_SIZE = 32


def _build_tiny_onnx(path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, IMAGE_SIZE, IMAGE_SIZE])
    out = helper.make_tensor_value_info("defect_probability", TensorProto.FLOAT, ["batch"])
    axes = helper.make_tensor("axes", TensorProto.INT64, [3], [1, 2, 3])
    nodes = [
        helper.make_node("ReduceMean", ["image", "axes"], ["m"], keepdims=0),
        helper.make_node("Sigmoid", ["m"], ["defect_probability"]),
    ]
    graph = helper.make_graph(nodes, "tiny", [inp], [out], initializer=[axes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, str(path))


@pytest.fixture
def tiny_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "model"
    bundle.mkdir()
    _build_tiny_onnx(bundle / "model.onnx")
    meta = {
        "model_version": "tiny-test",
        "backbone": "tiny",
        "class_names": ["normal", "defective"],
        "preprocess": {"image_size": IMAGE_SIZE, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "threshold": 0.5,
        "threshold_strategy": "fixed",
        "review_margin": 0.1,
        "model_sha256": sha256_file(bundle / "model.onnx"),
    }
    (bundle / "metadata.json").write_text(json.dumps(meta))
    return bundle


def png_bytes(color: tuple[int, int, int] = (128, 128, 128), size: int = 64, mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (size, size), color if mode == "RGB" else color[0]).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def bright_png() -> bytes:
    return png_bytes((255, 255, 255))  # normalised mean = +1 -> sigmoid(1) = 0.731 -> defective


@pytest.fixture
def dark_png() -> bytes:
    return png_bytes((0, 0, 0))  # normalised mean = -1 -> 0.269 -> normal


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
