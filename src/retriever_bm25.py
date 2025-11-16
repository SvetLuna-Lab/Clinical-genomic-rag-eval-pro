import math, os
from typing import Dict, List, Tuple

def _tok(s: str) -> List[str]:
    out = []
    for w in s.lower().split():
        t = "".join(ch for ch in w if ch.isalnum())
        if t: out.append(t)
    return out

class Doc:
    def __init__(self, doc_id: str, text: str) -> None:
        self.doc_id = doc_id
        self.text = text

class BM25Lite:
    def __init__(self, corpus_dir: str, k1: float=1.5, b: float=0.75) -> None:
        self.docs: List[Doc] = []
        for fn in os.listdir(corpus_dir):
            if not fn.endswith((".md", ".txt")): continue
            with open(os.path.join(corpus_dir, fn), "r", encoding="utf-8") as f:
                self.docs.append(Doc(fn, f.read()))
        self.N = len(self.docs)
        self.df: Dict[str, int] = {}
        self.len: Dict[str, int] = {}
        total = 0
        for d in self.docs:
            toks = _tok(d.text)
            total += len(toks)
            self.len[d.doc_id] = len(toks)
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.avg = total / self.N if self.N else 0
        self.k1, self.b = k1, b

    def score(self, q: str, d: Doc) -> float:
        if not self.N: return 0.0
        dt = _tok(d.text); qts = _tok(q)
        tf: Dict[str, int] = {}
        for t in dt: tf[t] = tf.get(t, 0) + 1
        s = 0.0; L = len(dt)
        for term in qts:
            if term not in tf: continue
            df = self.df.get(term, 0); 
            if df == 0: continue
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * L / (self.avg or 1.0))
            s += idf * (freq * (self.k1 + 1) / denom)
        return s

    def retrieve(self, q: str, top_k: int=5) -> List[Tuple[str,float]]:
        scored = [(d.doc_id, self.score(q, d)) for d in self.docs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
