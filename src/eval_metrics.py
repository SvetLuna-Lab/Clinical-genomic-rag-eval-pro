from typing import Dict, List, Set

def retrieval_hit_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int) -> float:
    return 1.0 if any(doc in gold_ids for doc in retrieved_ids[:k]) else 0.0

def citation_recall(answer_json: Dict, gold_ids: Set[str]) -> float:
    if not gold_ids: return 0.0
    cited = {c.get("doc_id") for c in answer_json.get("citations", []) if c.get("doc_id")}
    return len(gold_ids & cited) / len(gold_ids)

def keyword_coverage(claim: str, expected: List[str]) -> float:
    if not expected: return 0.0
    cl = claim.lower()
    hits = sum(1 for kw in expected if kw.lower() in cl)
    return hits / len(expected)

def _tok(s: str) -> List[str]:
    out = []
    for w in s.lower().split():
        t = "".join(ch for ch in w if ch.isalnum())
        if t: out.append(t)
    return out

def context_overlap(claim: str, ctx_text: str) -> float:
    ct = set(_tok(ctx_text)); cl = _tok(claim)
    if not cl: return 0.0
    hits = sum(1 for t in cl if t in ct)
    return hits / len(cl)

def faithfulness_stub(answer_json: Dict) -> float:
    # True if every citation shares at least one token with claim.
    claim = answer_json.get("claim","")
    claim_t = set(_tok(claim))
    cits = answer_json.get("citations", [])
    if not cits: return 0.0
    ok = 0
    for c in cits:
        qt = set(_tok(c.get("quote","")))
        ok += 1 if claim_t & qt else 0
    return ok / len(cits)
