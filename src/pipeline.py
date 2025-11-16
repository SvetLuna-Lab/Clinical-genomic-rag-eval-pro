import os, json
from typing import Dict, List, Tuple
from .retriever_bm25 import BM25Lite
from .chunking import section_aware_chunks
from .llm_client import LLMClient

def load_corpus(corpus_dir: str) -> Dict[str, str]:
    corpus: Dict[str,str] = {}
    for fn in os.listdir(corpus_dir):
        if fn.endswith((".md",".txt")):
            with open(os.path.join(corpus_dir, fn), "r", encoding="utf-8") as f:
                corpus[fn] = f.read()
    return corpus

def pick_quotes(doc_text: str, top_n: int=1) -> List[str]:
    # very naive: take first section/chunk
    ch = section_aware_chunks(doc_text)
    return ch[:top_n] if ch else [doc_text[:280]]

class Pipeline:
    def __init__(self, corpus_dir: str, retriever_kind: str="bm25", dense_model: str|None=None, hybrid_alpha: float=0.5):
        self.corpus = load_corpus(corpus_dir)
        self.retriever_kind = retriever_kind
        self.hybrid_alpha = hybrid_alpha
        self.dense_model = dense_model
        self.bm25 = BM25Lite(corpus_dir=corpus_dir)
        # dense retriever created lazily in run() if required

    def run(self, qid: str, question: str, top_k: int=5) -> Tuple[Dict, List[Dict]]:
        # retrieve
        if self.retriever_kind == "bm25":
            ranked = self.bm25.retrieve(question, top_k=top_k)
        elif self.retriever_kind in ("dense","hybrid"):
            from .retriever_dense import DenseRetriever
            from .retriever_hybrid import hybrid_merge
            dense = DenseRetriever(self.corpus, self.dense_model or "sentence-transformers/all-MiniLM-L6-v2")
            d_res = dense.retrieve(question, top_k=top_k)
            if self.retriever_kind == "dense":
                ranked = d_res
            else:
                b_res = self.bm25.retrieve(question, top_k=top_k)
                ranked = hybrid_merge(b_res, d_res, alpha=self.hybrid_alpha)[:top_k]
        else:
            raise ValueError(f"Unknown retriever kind: {self.retriever_kind}")

        contexts = []
        for doc_id, _ in ranked:
            quotes = pick_quotes(self.corpus[doc_id], top_n=1)
            contexts.append({"doc_id": doc_id, "quote": quotes[0] if quotes else ""})

        # answer (stub LLM client)
        client = LLMClient("stub")
        answer = client.answer(question, contexts, max_chars=280)
        answer["id"] = qid
        answer["metadata"]["retriever"] = self.retriever_kind
        return answer, contexts
