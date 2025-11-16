from typing import Dict, Tuple, List, Set
from .eval_metrics import _tok

def claim_evidence_pr(claim: str, citations: List[Dict]) -> Tuple[float,float,float]:
    """Precision/recall/F1 between claim tokens and concatenated citation tokens."""
    cl = set(_tok(claim))
    ev: Set[str] = set()
    for c in citations:
        ev |= set(_tok(c.get("quote","")))
    if not cl or not ev:
        return (0.0, 0.0, 0.0)
    tp = len(cl & ev)
    prec = tp / len(cl)
    rec  = tp / len(ev) if ev else 0.0
    f1 = (2*prec*rec)/(prec+rec) if (prec+rec)>0 else 0.0
    return (prec, rec, f1)
