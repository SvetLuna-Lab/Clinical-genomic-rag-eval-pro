from typing import Dict, List, Tuple

def hybrid_merge(bm25: List[Tuple[str,float]], dense: List[Tuple[str,float]], alpha: float=0.5) -> List[Tuple[str,float]]:
    """Linear score fusion: alpha * bm25 + (1-alpha) * dense; missing scores treated as 0."""
    scores: Dict[str, float] = {}
    for doc, s in bm25:
        scores[doc] = scores.get(doc, 0.0) + alpha * s
    for doc, s in dense:
        scores[doc] = scores.get(doc, 0.0) + (1.0 - alpha) * s
    out = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return out
