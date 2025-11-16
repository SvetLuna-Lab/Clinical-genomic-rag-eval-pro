import argparse, csv, json, os
from typing import Any, Dict, List, Set
from ..pipeline import Pipeline
from ..eval_metrics import retrieval_hit_at_k, citation_recall, keyword_coverage, context_overlap, faithfulness_stub
from ..faithfulness import claim_evidence_pr
from ..error_taxonomy import tag_errors
from ..utils.io import read_json, ensure_dir
from ..reporting.report_html import main as build_html

def load_cfg(path: str) -> Dict[str, Any]:
    import yaml
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def concat_citations(answer: Dict) -> str:
    return " ".join(c.get("quote","") for c in answer.get("citations", []))

def main(config_path: str) -> None:
    cfg = load_cfg(config_path)
    out_dir = cfg["paths"]["out_dir"]; ensure_dir(out_dir)
    data_dir = cfg["paths"]["data_dir"]; corpus_dir = cfg["paths"]["corpus_dir"]
    questions_path = cfg["paths"]["questions"]
    qset: List[Dict[str, Any]] = read_json(questions_path)

    pipe = Pipeline(
        corpus_dir=corpus_dir,
        retriever_kind=cfg["retrieval"]["kind"],
        hybrid_alpha=cfg["retrieval"].get("hybrid_alpha", 0.5),
        dense_model=cfg["retrieval"].get("model_name"),
    )
    top_k = int(cfg["retrieval"]["top_k"])
    hit_k = int(cfg["metrics"]["hit_k"])

    jsonl_path = os.path.join(out_dir, "eval_report.jsonl")
    csv_path   = os.path.join(out_dir, "eval_report.csv")
    html_path  = os.path.join(out_dir, "report.html")

    rows_csv: List[Dict[str, Any]] = []
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for q in qset:
            qid = q["id"]; question = q["question"]
            expected = q.get("expected_keywords", [])
            gold_ids: Set[str] = set(q.get("must_be_grounded_in", []))

            answer, ctxs = pipe.run(qid, question, top_k=top_k)
            retrieved_ids = [c["doc_id"] for c in ctxs]

            hitk = retrieval_hit_at_k(retrieved_ids, gold_ids, k=hit_k)
            citrec = citation_recall(answer, gold_ids)
            cov = keyword_coverage(answer.get("claim",""), expected)
            ovlp = context_overlap(answer.get("claim",""), concat_citations(answer))
            faith = faithfulness_stub(answer)
            prec, rec, f1 = claim_evidence_pr(answer.get("claim",""), answer.get("citations", []))
            metrics = {
                "hit@k": hitk,
                "citation_recall": citrec,
                "keyword_coverage": cov,
                "context_overlap": ovlp,
                "faithfulness_stub": faith,
                "faithfulness_precision": prec,
                "faithfulness_recall": rec,
                "faithfulness_f1": f1,
            }
            tags = tag_errors(metrics, cfg["metrics"].get("thresholds", {}))

            record = {"id": qid, "question": question, "answer": answer,
                      "retrieved_doc_ids": retrieved_ids, "metrics": metrics, "tags": tags}
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

            row = {"id": qid, **metrics}
            rows_csv.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        import csv as _csv
        writer = _csv.DictWriter(cf, fieldnames=list(rows_csv[0].keys()))
        writer.writeheader(); writer.writerows(rows_csv)

    # quick HTML
    build_html(jsonl_path, html_path)
    print("=== Pro RAG evaluation complete ===")
    print(f"JSONL: {os.path.abspath(jsonl_path)}")
    print(f"CSV:   {os.path.abspath(csv_path)}")
    print(f"HTML:  {os.path.abspath(html_path)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    a = ap.parse_args()
    main(a.config)
