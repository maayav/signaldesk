PYTHON ?= .venv/bin/python

.PHONY: install data train api ui test lint check docker
install:
	python3 -m venv .venv
	PIP_USER=0 $(PYTHON) -m pip install -r requirements-lock.txt
	PIP_USER=0 $(PYTHON) -m pip install --no-deps -e .
data:
	$(PYTHON) -m signaldesk.data --data-dir data/raw
train:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $(PYTHON) -m signaldesk.train --data-dir data/raw --artifact-dir artifacts
api:
	$(PYTHON) -m uvicorn signaldesk.api:app --host 127.0.0.1 --port 8000
ui:
	$(PYTHON) -m streamlit run app/dashboard.py --browser.gatherUsageStats=false
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check src app tests scripts
check: lint test
docker:
	docker compose up --build
