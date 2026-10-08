"""Call a running API with the example images and record the responses.

    python scripts/demo_client.py --url http://localhost:8000 --out examples/example_predictions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--images", default="examples/images")
    ap.add_argument("--out", default="examples/example_predictions.json")
    args = ap.parse_args()

    files = sorted(Path(args.images).glob("*.png"))
    results = []
    with httpx.Client(base_url=args.url, timeout=30) as client:
        print(client.get("/v1/model").json())
        for f in files:
            r = client.post("/v1/predict", files={"file": (f.name, f.read_bytes(), "image/png")})
            body = r.json()
            results.append({"file": f.name, "status": r.status_code, **body})
            p = body.get("prediction", {})
            print(f"{f.name:<32} -> {p.get('predicted_class'):<9} conf={p.get('confidence', 0):.3f} "
                  f"P(defect)={p.get('defect_probability', 0):.3f} review={p.get('review_recommended')} "
                  f"total={body.get('timing', {}).get('total_ms')}ms")
        # Error-handling examples
        bad = client.post("/v1/predict", files={"file": ("corrupt.png", b"not an image", "image/png")})
        results.append({"file": "corrupt.png", "status": bad.status_code, **bad.json()})
        print("corrupt.png ->", bad.status_code, bad.json()["error"])
        batch = client.post("/v1/predict/batch",
                            files=[("files", (f.name, f.read_bytes(), "image/png")) for f in files[:4]])
        results.append({"file": "batch(4)", "status": batch.status_code, **batch.json()})
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
