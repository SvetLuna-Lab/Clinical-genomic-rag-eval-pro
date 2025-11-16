from dataclasses import dataclass
from typing import Any, Dict, List, Iterable

# ---------------------------
# Tokenization helpers
# ---------------------------

def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in text.lower().split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if token:
            tokens.append(token)
    return tokens

def _as_text(answer: Any) -> str:
    """
    Normalize answer to plain text.
    - str -> as is
    - dict -> try ["text","answer","output","content"]; else JSON-dump; else str(value)
    """
    if isinstance(answer, str):
        return answer
    if isinstance(answer, dict):
        for k in ("text", "answer", "output", "content"):
            v = answer.get(k)
            if isinstance(v, str):
                return v
        try:
            import json
            return json.dumps(answer, ensure_ascii=False)
        except Exception:
            return str(answer)
    return str(answer)

# Backward-compat alias used by faithfulness.py tests
def _tok(text: str) -> List[str]:  # noqa: N802 (test expects this exact name)
    return _tokenize(text)

# ---------------------------
# Core metrics
# ---------------------------

def keyword_coverage(answer: Any, expected_keywords: List[str]) -> float:
    """
    Fraction of expected keywords found in the answer (case-insensitive substring).
    'answer' may be str or dict with {text|answer|output|content}.
    """
    if not expected_keywords:
        return 0.0
    ans_l = _as_text(answer).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in ans_l)
    return hits / len(expected_keywords)

def context_overlap(answer: Any, context_text: str) -> float:
    """
    Simple grounding proxy:
    ratio of answer tokens that also appear in the provided reference context.
    """
    ans_tokens = _tokenize(_as_text(answer))
    ctx_tokens = set(_tokenize(context_text))
    if not ans_tokens:
        return 0.0
    hits = sum(1 for t in ans_tokens if t in ctx_tokens)
    return hits / len(ans_tokens)

def retrieval_hit_at_k(
    retrieved: Iterable[Any],
    gold_doc_ids: Iterable[str],
    k: int = 3,
) -> float:
    """
    Hit@k: 1.0 if any of the top-k retrieved doc_ids is in gold_doc_ids, else 0.0.

    'retrieved' may be:
      - list of dicts with key 'doc_id'
      - list/tuple of (doc, score) where doc has attribute 'doc_id' or dict with 'doc_id'
      - direct list of strings (doc_ids)
    """
    # normalize retrieved -> list[str]
    ids: List[str] = []
    for i, item in enumerate(retrieved):
        if i >= k:
            break
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict):
            did = item.get("doc_id")
            if isinstance(did, str):
                ids.append(did)
        elif isinstance(item, (list, tuple)) and item:
            doc = item[0]
            if isinstance(doc, dict):
                did = doc.get("doc_id")
                if isinstance(did, str):
                    ids.append(did)
            else:
                # object with attribute doc_id
                did = getattr(doc, "doc_id", None)
                if isinstance(did, str):
                    ids.append(did)
        else:
            did = getattr(item, "doc_id", None)
            if isinstance(did, str):
                ids.append(did)

    gold = set(gold_doc_ids or [])
    return 1.0 if any(d in gold for d in ids) else 0.0

# ---------------------------
# Aggregation helpers (used by runner/tests)
# ---------------------------

@dataclass
class EvalResult:
    question_id: str
    coverage: float
    overlap: float
    score: float

def compute_score(coverage: float, overlap: float, alpha: float = 0.5) -> float:
    """Final score = alpha * coverage + (1 - alpha) * overlap."""
    return alpha * coverage + (1.0 - alpha) * overlap

def evaluate_single(
    question_id: str,
    answer: Any,
    expected_keywords: List[str],
    context_text: str,
    alpha: float = 0.5,
) -> EvalResult:
    cov = keyword_coverage(answer, expected_keywords)
    ov  = context_overlap(answer, context_text)
    sc  = compute_score(cov, ov, alpha=alpha)
    return EvalResult(question_id=question_id, coverage=cov, overlap=ov, score=sc)

def to_dict(res: EvalResult) -> Dict[str, float]:
    return {
        "question_id": res.question_id,
        "keyword_coverage": res.coverage,
        "context_overlap": res.overlap,
        "score": res.score,
    }
