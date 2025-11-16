"""
Run clinical/genomic RAG evaluation.

- Loads YAML config (paths, retrieval/generation, metrics).
- Optional --bm25 flag to force BM25 (для CI smoke без emb deps).
- Прогоняет вопросы из data/eval_questions.json.
- Пишет артефакты в reports/:
    * eval_report.jsonl
    * eval_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

# ---- pipeline import (be tolerant to naming/signatures) -----------------------
try:
    from src.pipeline import Pipeline as _Pipeline
except Exception:
    from src.pipeline import RagPipeline as _Pipeline  # type: ignore

from src.eval_metrics import evaluate_single, to_dict


# ---- small utils --------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_auto_to_json(r), ensure_ascii=False))
            f.write("\n")


def _write_csv(path: Path, rows: List[Dict[str, Any]], field_order: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in field_order})


def _auto_to_json(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _auto_to_json(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _auto_to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_auto_to_json(v) for v in obj]
    return obj


def _concat_docs(corpus_dir: Path, doc_ids: List[str]) -> str:
    parts: List[str] = []
    for doc_id in doc_ids:
        p = corpus_dir / doc_id
        if p.exists():
            parts.append(p.read_text(encoding="utf-8"))
    return "\n".join(parts)


# ---- CLI ----------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RAG evaluation")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    p.add_argument("--out_jsonl", type=str, default=None, help="Override output JSONL path.")
    p.add_argument("--out_csv", type=str, default=None, help="Override output CSV path.")
    p.add_argument("--top_k", type=int, default=None, help="Override retrieval.top_k.")
    p.add_argument("--bm25", action="store_true", help="Force BM25 for CI smoke.")
    return p.parse_args(argv)


# ---- main ---------------------------------------------------------------------

def main(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    args = parse_args(None)

    if args.top_k is not None:
        cfg.setdefault("retrieval", {})["top_k"] = int(args.top_k)
    if args.bm25:
        cfg.setdefault("retrieval", {})["kind"] = "bm25"

    paths = cfg.get("paths", {})
    data_dir = Path(paths.get("data_dir", "data"))
    corpus_dir = Path(paths.get("corpus_dir", str(data_dir / "corpus")))
    q_path = Path(paths.get("questions", str(data_dir / "eval_questions.json")))
    out_dir = Path(paths.get("out_dir", "reports"))
    _ensure_dir(out_dir)

    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else out_dir / "eval_report.jsonl"
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / "eval_report.csv"

    # --- construct pipeline (handle different __init__ signatures gracefully)
    pipe = None
    init_err: Exception | None = None
    for kwargs in (
        {"corpus_dir": str(corpus_dir), "config": cfg},
        {"corpus_dir": str(corpus_dir)},                 # demo-стиль
        {"corpus_dir": corpus_dir},                      # Path variant
    ):
        try:
            pipe = _Pipeline(**kwargs)  # type: ignore[arg-type]
            init_err = None
            break
        except TypeError as e:
            init_err = e
            continue
    if pipe is None:
        raise RuntimeError(f"Cannot construct pipeline with known signatures: {init_err}")

    questions: List[Dict[str, Any]] = _load_json(q_path)

    results_jsonl: List[Dict[str, Any]] = []
    results_csv_rows: List[Dict[str, Any]] = []
    top_k = int(cfg.get("retrieval", {}).get("top_k", 5))

    for q in questions:
        qid = q["id"]
        question = q["question"]
        expected_keywords = q.get("expected_keywords", [])
        must_be_grounded_in = q.get("must_be_grounded_in", [])

        # --- run pipeline (support different run signatures)
        try:
            answer, contexts = pipe.run(qid=qid, question=question, top_k=top_k)  # type: ignore[attr-defined]
        except TypeError:
            answer, contexts = pipe.run(question=question, top_k=top_k)  # type: ignore[misc]
        except Exception:
            # very old stub: run(question, top_k)
            answer, contexts = pipe.run(question, top_k)  # type: ignore[misc]

        gold_context = _concat_docs(corpus_dir, must_be_grounded_in)

        eval_res = evaluate_single(
            question_id=qid,
            answer=answer,
            expected_keywords=expected_keywords,
            context_text=gold_context,
            alpha=float(cfg.get("metrics", {}).get("alpha", 0.5)),
        )

        record = {
            "id": qid,
            "question": question,
            "answer": answer,
            "expected_keywords": expected_keywords,
            "must_be_grounded_in": must_be_grounded_in,
            "metrics": to_dict(eval_res),
            "retrieved_docs": [
                {
                    "doc_id": getattr(ctx, "doc_id", None),
                    "score": getattr(ctx, "score", None),
                }
                for ctx in (contexts or [])
            ],
        }
        results_jsonl.append(record)

        flat = {"id": qid}
        flat.update(to_dict(eval_res))
        results_csv_rows.append(flat)

    _write_jsonl(out_jsonl, results_jsonl)
    _write_csv(out_csv, results_csv_rows, ["id", "keyword_coverage", "context_overlap", "score"])

    print("=== Pro RAG evaluation complete ===")
    print(f"JSONL: {out_jsonl}")
    print(f"CSV:   {out_csv}")
    return {"jsonl": str(out_jsonl), "csv": str(out_csv), "n": len(results_jsonl)}


if __name__ == "__main__":
    ns = parse_args()
    sys.exit(0 if main(ns.config) else 1)
