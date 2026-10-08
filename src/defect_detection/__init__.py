"""Visual defect detection: dataset tooling, training, evaluation and ONNX serving."""

__version__ = "1.0.0"

CLASS_NAMES: tuple[str, str] = ("normal", "defective")
POSITIVE_CLASS = "defective"
