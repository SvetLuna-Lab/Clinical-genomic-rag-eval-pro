# src/eval_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


def _tokenize(text: str) -> List[str]:
    # Very simple tokenizer: lowercase, split on whitespace, keep only [a-z0-9]
    tokens = []
    for raw in text.lower().split():
        tok = "".join(ch for ch in raw if ch.isalnum())
        if tok:
            tokens.append(tok)
    return tokens


def keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    """
    Share of expected keywords that actually appear in the answer (substring, case-insensitive).
    """
    if not expected_keywords:
        return 0.0
    ans_l = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in ans_l)
    return hits / len(expected_keywords)


def context_overlap(answer: str, context_text: str) -> float:
    """
    Very simple "non-hallucination" proxy:
    fraction of answer tokens that appear in the provided context.
    """
    ans_toks = _tokenize(answer)
    if not ans_toks:
        return 0.0
    ctx_toks = set(_tokenize(context_text))
    hits = sum(1 for t in ans_toks if t in ctx_toks)
    return hits / len(ans_toks)


@dataclass
class EvalResult:
    question_id: str
    coverage: float
    overlap: float
    score: float


def compute_score(coverage: float, overlap: float, alpha: float = 0.5) -> float:
    """
    Combined score = alpha * coverage + (1 - alpha) * overlap.
    """
    return alpha * coverage + (1.0 - alpha) * overlap


def evaluate_single(
    question_id: str,
    answer: str,
    expected_keywords: List[str],
    context_text: str,
    alpha: float = 0.5,
) -> EvalResult:
    """
    Thin wrapper used by runners:
    computes coverage/overlap and combines them into a scalar score.
    """
    cov = keyword_coverage(answer, expected_keywords)
    ov = context_overlap(answer, context_text)
    sc = compute_score(cov, ov, alpha=alpha)
    return EvalResult(question_id=question_id, coverage=cov, overlap=ov, score=sc)


def to_dict(res: EvalResult) -> Dict[str, float]:
    return {
        "question_id": res.question_id,
        "keyword_coverage": res.coverage,
        "context_overlap": res.overlap,
        "score": res.score,
    }


# Optional explicit export list (helps linters and star-imports)
__all__ = [
    "keyword_coverage",
    "context_overlap",
    "compute_score",
    "EvalResult",
    "evaluate_single",
    "to_dict",
]

