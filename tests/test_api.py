import pytest
from fastapi.testclient import TestClient

from defect_detection.serving.app import create_app
from defect_detection.serving.settings import Settings
from tests.conftest import png_bytes


@pytest.fixture
def make_client(tiny_bundle):
    def _make(**overrides):
        settings = Settings(model_dir=tiny_bundle, json_logs=False, **overrides)
        return TestClient(create_app(settings))

    return _make


@pytest.fixture
def client(make_client):
    with make_client() as c:
        yield c


def test_health_and_model_info(client):
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").json()["model_version"] == "tiny-test"
    info = client.get("/v1/model").json()
    assert info["class_names"] == ["normal", "defective"] and info["threshold"] == 0.5


def test_predict_returns_class_and_confidence(client, bright_png):
    r = client.post("/v1/predict", files={"file": ("part.png", bright_png, "image/png")},
                    headers={"X-Request-ID": "abc123"})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"]["predicted_class"] == "defective"
    assert 0.5 <= body["prediction"]["confidence"] <= 1
    assert body["request_id"] == "abc123" and r.headers["X-Request-ID"] == "abc123"
    assert body["model_version"] == "tiny-test"
    assert set(body["timing"]) == {"decode_ms", "preprocess_ms", "inference_ms", "total_ms"}


def test_predict_normal(client, dark_png):
    r = client.post("/v1/predict", files={"file": ("p.png", dark_png, "image/png")})
    assert r.json()["prediction"]["predicted_class"] == "normal"


@pytest.mark.parametrize(
    ("payload", "ctype", "status", "code"),
    [
        (b"garbage", "image/png", 422, "invalid_image"),
        (b"hello", "text/plain", 415, "unsupported_media_type"),
        (png_bytes(size=8), "image/png", 422, "invalid_image"),
    ],
    ids=["corrupt", "wrong-type", "too-small"],
)
def test_predict_rejects_bad_input(client, payload, ctype, status, code):
    r = client.post("/v1/predict", files={"file": ("x", payload, ctype)})
    assert r.status_code == status
    assert r.json()["error"]["code"] == code
    assert r.json()["request_id"]


def test_missing_file_is_422(client):
    r = client.post("/v1/predict")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_upload_size_limit(make_client):
    with make_client(max_upload_bytes=1000) as c:
        r = c.post("/v1/predict", files={"file": ("big.png", png_bytes(size=512), "image/png")})
        assert r.status_code == 413


def test_batch_reports_errors_per_item(client, bright_png, dark_png):
    files = [("files", ("a.png", bright_png, "image/png")),
             ("files", ("bad.png", b"nope", "image/png")),
             ("files", ("b.png", dark_png, "image/png"))]
    r = client.post("/v1/predict/batch", files=files)
    assert r.status_code == 200
    res = r.json()["results"]
    assert [x["filename"] for x in res] == ["a.png", "bad.png", "b.png"]
    assert res[0]["prediction"]["predicted_class"] == "defective"
    assert res[1]["error"]["code"] == "invalid_image" and res[1]["prediction"] is None
    assert res[2]["prediction"]["predicted_class"] == "normal"


def test_batch_size_limit(make_client, dark_png):
    with make_client(max_batch_size=2) as c:
        files = [("files", (f"{i}.png", dark_png, "image/png")) for i in range(3)]
        assert c.post("/v1/predict/batch", files=files).status_code == 413


def test_api_key_enforced(make_client, dark_png):
    with make_client(api_key="s3cret") as c:
        f = {"file": ("p.png", dark_png, "image/png")}
        assert c.post("/v1/predict", files=f).status_code == 401
        assert c.post("/v1/predict", files=f, headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/health/live").status_code == 200  # probes stay open


def test_metrics_exposed(client, dark_png):
    client.post("/v1/predict", files={"file": ("p.png", dark_png, "image/png")})
    text = client.get("/metrics").text
    assert 'vdd_predictions_total{predicted_class="normal"} 1.0' in text
    assert "vdd_inference_seconds_bucket" in text
    assert 'vdd_model_info{model_version="tiny-test"' in text


def test_startup_fails_fast_without_model(tmp_path):
    app = create_app(Settings(model_dir=tmp_path, json_logs=False))
    with pytest.raises(Exception, match="bundle incomplete"), TestClient(app):
        pass
