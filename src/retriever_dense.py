# Optional dense retriever: Sentence-Transformers + FAISS (CPU)
from typing import List, Tuple, Dict
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception as e:  # graceful import error for environments without deps
    SentenceTransformer = None
    faiss = None

class DenseRetriever:
    def __init__(self, corpus: Dict[str,str], model_name: str) -> None:
        if SentenceTransformer is None or faiss is None:
            raise RuntimeError("Dense retriever requires sentence-transformers and faiss-cpu.")
        self.ids = list(corpus.keys())
        self.texts = [corpus[i] for i in self.ids]
        self.model = SentenceTransformer(model_name)
        self.emb = self.model.encode(self.texts, convert_to_numpy=True, show_progress_bar=False)
        d = self.emb.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(self.emb)
        self.index.add(self.emb)

    def retrieve(self, q: str, top_k: int=5) -> List[Tuple[str,float]]:
        qe = self.model.encode([q], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(qe)
        D, I = self.index.search(qe, top_k)
        return [(self.ids[i], float(D[0, j])) for j, i in enumerate(I[0])]
