PY=python

.PHONY: setup fetch train evaluate app report test clean

setup:
	$(PY) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

fetch:
	$(PY) src/data_prep.py --fetch --out data/raw --hash artifacts/data_hash.json

train:
	$(PY) src/model_train.py --config configs/train.yaml

evaluate:
	$(PY) src/evaluate.py --config configs/train.yaml

app:
	@echo "Open notebooks/app.ipynb in Deepnote or Jupyter to run the interactive app."

report:
	$(PY) src/reporting/build_report.py --metrics artifacts/metrics.json --out reports/findings.md

test:
	. .venv/bin/activate && pytest -q

clean:
	rm -rf .venv __pycache__ .pytest_cache artifacts/* data/processed/*
