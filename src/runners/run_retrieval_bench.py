import argparse, os, json
from typing import Any, Dict, List
from ..retriever_bm25 import BM25Lite
from ..pipeline import load_corpus

def load_cfg(path: str) -> Dict[str, Any]:
    import yaml
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def main(config_path: str) -> None:
    cfg = load_cfg(config_path)
    corpus_dir = cfg["paths"]["corpus_dir"]
    questions_path = cfg["paths"]["questions"]
    with open(questions_path,"r",encoding="utf-8") as f:
        qset: List[Dict[str,Any]] = json.load(f)

    bm25 = BM25Lite(corpus_dir)
    hit_at_5 = 0; n = 0
    for q in qset:
        gold = set(q.get("must_be_grounded_in", []))
        r = bm25.retrieve(q["question"], top_k=5)
        ids = [d for d,_ in r]
        hit_at_5 += 1 if any(i in gold for i in ids[:5]) else 0
        n += 1
    avg = hit_at_5 / n if n else 0.0
    print(f"BM25 hit@5: {avg:.3f} ({hit_at_5}/{n})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    a = p.parse_args()
    main(a.config)
