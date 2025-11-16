import os, sys, subprocess

def test_run_eval_smoke(tmp_path):
    cmd = [sys.executable, "-m", "src.runners.run_eval", "--config", "configs/default.yaml"]
    res = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True)
    assert res.returncode == 0
    assert os.path.exists("reports/eval_report.jsonl")
    assert os.path.exists("reports/eval_report.csv")
    assert os.path.exists("reports/report.html")
