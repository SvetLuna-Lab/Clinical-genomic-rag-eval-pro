from typing import List

SECTION_MARKERS = ("## HPI", "## Assessment and Plan", "## Pathology")

def section_aware_chunks(text: str) -> List[str]:
    """Very small section-aware splitter: split on known markers, else return whole."""
    parts: List[str] = []
    buf: List[str] = []
    for line in text.splitlines():
        if any(line.strip().startswith(m) for m in SECTION_MARKERS):
            if buf:
                parts.append("\n".join(buf).strip())
                buf = []
        buf.append(line)
    if buf:
        parts.append("\n".join(buf).strip())
    return [p for p in parts if p]
