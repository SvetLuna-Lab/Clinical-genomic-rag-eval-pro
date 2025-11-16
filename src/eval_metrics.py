from dataclasses import dataclass
from typing import Dict, List, Any

def _tokenize(text: str) -> List[str]:
    tokens = []
    for raw in text.lower().split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if token:
            tokens.append(token)
    return tokens

def _as_text(answer: Any) -> str:
    """
    Normalize an answer to plain text:
    - str -> as is
    - dict -> try keys ["text", "answer", "output", "content"], else compact JSON
    - other -> str(value)
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

def keyword_coverage(answer: Any, expected_keywords: List[str]) -> float:
    """
    Fraction of expected keywords actually mentioned in the answer.
    Case-insensitive substring check; 'answer' may be str or dict.
    """
    if not expected_keywords:
        return 0.0

    ans_l = _as_text(answer).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in ans_l)
    return hits / len(expected_keywords)

def context_overlap(answer: Any, context_text: str) -> float:
    """
    Simple non-hallucination proxy:
    ratio of answer tokens that appear in the reference context.
    """
    ans_tokens = _tokenize(_as_text(answer))
    ctx_tokens = set(_tokenize(context_text))

    if not ans_tokens:
        return 0.0

    hits = sum(1 for t in ans_tokens if t in ctx_tokens)
    return hits / len(ans_tokens)

@dataclass
class EvalResult:
    question_id: str
    coverage: float
    overlap: float
    score: float

def compute_score(coverage: float, overlap: float, alpha: float = 0.5) -> float:
    """Final score = alpha*coverage + (1-alpha)*overlap."""
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
    score = compute_score(cov, ov, alpha=alpha)
    return EvalResult(question_id=question_id, coverage=cov, overlap=ov, score=score)

def to_dict(res: EvalResult) -> Dict[str, float]:
    return {
        "question_id": res.question_id,
        "keyword_coverage": res.coverage,
        "context_overlap": res.overlap,
        "score": res.score,
    }
