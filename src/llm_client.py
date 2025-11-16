from typing import Dict, List

class LLMClient:
    """Stub adapter; replace with OpenAI/Groq/etc when enabled."""
    def __init__(self, model_name: str = "stub"):
        self.model_name = model_name

    def answer(self, question: str, contexts: List[Dict[str, str]], max_chars: int=280) -> Dict:
        # naive templated answer
        ctx_text = " ".join(c["quote"] for c in contexts if c.get("quote"))
        claim = (ctx_text[:max_chars] + "...") if len(ctx_text) > max_chars else ctx_text
        return {
            "claim": claim or "No grounded answer.",
            "citations": contexts,
            "metadata": {"model": self.model_name}
        }
