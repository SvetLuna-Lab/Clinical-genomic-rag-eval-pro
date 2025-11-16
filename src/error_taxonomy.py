from typing import Dict

def tag_errors(metrics: Dict[str,float], thresholds: Dict[str,float]) -> Dict[str,bool]:
    """Simple rule-based tags based on metric thresholds."""
    tags = {}
    tags["no_hit_at_k"] = (metrics.get("hit@k",0.0) < 1.0)
    tags["low_coverage"] = (metrics.get("keyword_coverage",0.0) < thresholds.get("low_coverage",0.4))
    tags["low_overlap"] = (metrics.get("context_overlap",0.0) < thresholds.get("low_overlap",0.5))
    tags["no_citations"] = (metrics.get("citation_recall",0.0) == 0.0)
    return tags
