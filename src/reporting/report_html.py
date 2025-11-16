import argparse, json, os
from typing import Dict, List

TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>RAG Eval Report</title>
<style>body{font-family:system-ui;margin:24px} table{border-collapse:collapse}
td,th{border:1px solid #ccc;padding:6px 8px} th{background:#f5f5f5}</style></head>
<body>
<h1>RAG Evaluation Summary</h1>
<p><b>Records:</b> {n}</p>
<table><thead><tr>
<th>ID</th><th>hit@k</th><th>citation_recall</th><th>keyword_coverage</th><th>context_overlap</th><th>faithfulness_stub</th>
</tr></thead><tbody>
{rows}
</tbody></table>
</body></html>"""

def main(in_jsonl: str, out_html: str) -> None:
    rows_html: List[str] = []
    n = 0
    with open(in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            m: Dict = r.get("metrics", {})
            rows_html.append(
                f"<tr><td>{r.get('id')}</td><td>{m.get('hit@k',0):.2f}</td>"
                f"<td>{m.get('citation_recall',0):.2f}</td>"
                f"<td>{m.get('keyword_coverage',0):.2f}</td>"
                f"<td>{m.get('context_overlap',0):.2f}</td>"
                f"<td>{m.get('faithfulness_stub',0):.2f}</td></tr>"
            )
            n += 1
    html = TEMPLATE.format(n=n, rows="\n".join(rows_html))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML saved: {os.path.abspath(out_html)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_jsonl", required=True)
    p.add_argument("--out_html", required=True)
    a = p.parse_args()
    main(a.in_jsonl, a.out_html)
