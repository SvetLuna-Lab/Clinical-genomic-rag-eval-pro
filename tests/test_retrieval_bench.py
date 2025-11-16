import subprocess, sys, os

def test_bench_runs():
    cmd = [sys.executable, "-m", "src.runners.run_retrieval_bench", "--config", "configs/default.yaml"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "BM25 hit@5:" in res.stdout
