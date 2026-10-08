PY ?= python
CONFIG ?= configs/train.yaml

.PHONY: install data train compare promote test lint serve docker-build docker-run

install:            ## dev install (CPU torch)
	$(PY) -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/dev.txt
	$(PY) -m pip install --no-deps -e .

data:               ## download public dataset, build manifest, EDA, split
	vdd download && vdd prepare --config $(CONFIG)

train:              ## train -> export -> evaluate -> error analysis -> benchmark
	vdd pipeline --config $(CONFIG)

compare:            ## backbone accuracy / latency comparison
	vdd compare --config $(CONFIG)

promote:            ## publish the configured run as the served model
	vdd promote --config $(CONFIG)

test:
	pytest

lint:
	ruff check src tests

serve:
	vdd serve --port 8000

docker-build:
	docker build -t defect-detection-api:latest .

docker-run:
	docker run --rm -p 8000:8000 defect-detection-api:latest
