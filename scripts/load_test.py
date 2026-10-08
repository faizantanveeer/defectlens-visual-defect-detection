"""Minimal concurrent load test for /v1/predict.

    python scripts/load_test.py --url http://localhost:8000 --image examples/images/normal_example.png \
        --concurrency 8 --requests 400
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np


async def worker(client: httpx.AsyncClient, payload: bytes, queue: asyncio.Queue, lat: list, codes: Counter):
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        t = time.perf_counter()
        r = await client.post("/v1/predict", files={"file": ("x.png", payload, "image/png")})
        lat.append((time.perf_counter() - t) * 1000)
        codes[r.status_code] += 1


async def run(args) -> dict:
    payload = Path(args.image).read_bytes()
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(args.requests):
        queue.put_nowait(1)
    lat: list[float] = []
    codes: Counter = Counter()
    limits = httpx.Limits(max_connections=args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, timeout=60, limits=limits) as client:
        for _ in range(10):  # warm-up
            await client.post("/v1/predict", files={"file": ("x.png", payload, "image/png")})
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(client, payload, queue, lat, codes) for _ in range(args.concurrency)))
        wall = time.perf_counter() - t0
    a = np.asarray(lat)
    return {
        "concurrency": args.concurrency,
        "requests": args.requests,
        "throughput_rps": round(args.requests / wall, 1),
        "p50_ms": round(float(np.percentile(a, 50)), 1),
        "p95_ms": round(float(np.percentile(a, 95)), 1),
        "p99_ms": round(float(np.percentile(a, 99)), 1),
        "status_codes": dict(codes),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--image", default="examples/images/normal_example.png")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=400)
    print(json.dumps(asyncio.run(run(ap.parse_args())), indent=2))


if __name__ == "__main__":
    main()
