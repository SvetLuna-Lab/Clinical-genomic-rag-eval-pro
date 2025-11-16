#!/usr/bin/env bash
set -e
python -m src.runners.run_eval --config configs/default.yaml
python -m src.reporting.report_html --in_jsonl reports/eval_report.jsonl --out_html reports/report.html
echo "Artifacts in reports/"
