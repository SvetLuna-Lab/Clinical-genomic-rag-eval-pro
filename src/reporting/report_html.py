from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def render_html(rows: List[Dict[str, Any]]) -> str:
    # compute aggregates
    scores = [r.get("metrics", {}).get("score") for r in rows if r.get("metrics")]
    covs   = [r.get("metrics", {}).get("keyword_coverage") for r in rows if r.get("metrics")]
    ovs    = [r.get("metrics", {}).get("context_overlap") for r in rows if r.get("metrics")]
    avg_score = mean(scores) if scores else 0.0
    avg_cov   = mean(covs)   if covs   else 0.0
    avg_ov    = mean(ovs)    if ovs    else 0.0

    # table rows
    tr_html: List[str] = []
    for r in rows:
        m = r.get("metrics", {})
        tr_html.append(
            "<tr>"
            f"<td>{r.get('id','')}</td>"
            f"<td>{(m.get('score', 0.0)):.3f}</td>"
            f"<td>{(m.get('keyword_coverage', 0.0)):.3f}</td>"
            f"<td>{(m.get('context_overlap', 0.0)):.3f}</td>"
            f"<td>{(r.get('answer_text','') or '')[:120].replace('<','&lt;')}</td>"
            "</tr>"
        )

    # full HTML (f-string; CSS braces are safe)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Clinical+Genomic RAG — Evaluation Report</title>
  <style>
    :root {{
      --bg: #0b1220; --fg: #e9eefb; --muted: #9db0cf; --accent: #6cc4ff; --row: #121b2e;
    }}
    body {{
      margin: 0; padding: 24px; background: var(--bg); color: var(--fg);
      font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    }}
    h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
    .summary {{ margin: 6px 0 18px 0; color: var(--muted); }}
    table {{
      width: 100%; border-collapse: collapse; background: #091126; border-radius: 10px; overflow: hidden;
    }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #1b2a4a; }}
    tr:nth-child(even) {{ background: var(--row); }}
    th {{ color: var(--accent); font-weight: 600; }}
    .kpi {{ display: inline-block; margin-right: 16px; }}
    .kpi b {{ color: var(--fg); }}
  </style>
</head>
<body>
  <h1>Clinical+Genomic RAG — Evaluation Report</h1>
  <div class="summary">
    <span class="kpi"><b>Items:</b> {len(rows)}</span>
    <span class="kpi"><b>avg score:</b> {avg_score:.3f}</span>
    <span class="kpi"><b>avg coverage:</b> {avg_cov:.3f}</span>
    <span class="kpi"><b>avg overlap:</b> {avg_ov:.3f}</span>
  </div>

  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Score</th>
        <th>Coverage</th>
        <th>Overlap</th>
        <th>Answer (preview)</th>
      </tr>
    </thead>
    <tbody>
      {''.join(tr_html)}
    </tbody>
  </table>
</body>
</html>"""
    return html

def main(in_jsonl: str, out_html: str) -> None:
    rows = load_jsonl(Path(in_jsonl))
    html = render_html(rows)
    Path(out_html).write_text(html, encoding="utf-8")
    print(f"HTML report written to {out_html}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True, help="Path to reports/eval_report.jsonl")
    ap.add_argument("--out_html", required=True, help="Path to write HTML summary")
    args = ap.parse_args()
    main(args.in_jsonl, args.out_html)

