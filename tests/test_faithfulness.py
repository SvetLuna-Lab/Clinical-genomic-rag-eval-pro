from src.faithfulness import claim_evidence_pr

def test_claim_evidence_f1():
    claim = "endocrine therapy recommended"
    cites = [{"quote":"adjuvant endocrine therapy is recommended"}]
    p,r,f1 = claim_evidence_pr(claim, cites)
    assert f1 > 0.0
