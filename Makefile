PY=python

.PHONY: setup test eval report bench clean

setup:
	$(PY) -m pip install -r requirements.txt

setup-emb:
	$(PY) -m pip install -r requirements-emb.txt

setup-llm:
	$(PY) -m pip install -r requirements-llm.txt

test:
	$(PY) -m pytest -q

eval:
	$(PY) -m src.runners.run_eval --config configs/default.yaml

report:
	$(PY) -m src.reporting.report_html --in_jsonl reports/eval_report.jsonl --out_html reports/report.html

bench:
	$(PY) -m src.runners.run_retrieval_bench --config configs/retrievers.yaml

clean:
	rm -f reports/*.jsonl reports/*.csv reports/*.html reports/*.png || true
