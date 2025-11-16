from src.eval_metrics import keyword_coverage, context_overlap, retrieval_hit_at_k

def test_keyword_coverage_basic():
    assert keyword_coverage("endocrine therapy is recommended", ["endocrine","adjuvant"]) == 0.5

def test_context_overlap_nonzero():
    claim = "pi3k inhibitors for pik3ca mutations"
    ctx = "PIK3CA mutations may respond to PI3K inhibitors"
    assert context_overlap(claim, ctx) > 0.0

def test_hit_at_k():
    assert retrieval_hit_at_k(["a","b","c"], {"x","b"}, k=3) == 1.0
