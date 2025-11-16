
# Clinical-Genomic RAG Eval Pro

Pro-grade **RAG evaluation** for clinical/genomic scenarios: hybrid retrieval (BM25 + embeddings), transparent faithfulness metrics, YAML configs, and HTML reports.  
**Python 3.10+**

---

## Highlights

- **Hybrid retrieval**: BM25 (pure Python) + optional dense retriever (Sentence-Transformers + FAISS).
- **Faithfulness+**: token-level claim↔evidence Precision/Recall/F1; citation recall; keyword coverage; context overlap.
- **Reproducible runs**: YAML configs, Makefile targets, CI, deterministic seeds.
- **Audit artifacts**: JSONL & CSV per question + HTML summary dashboard.
- **Small, extensible**: clean Python modules; easy to plug a real LLM client.

---

## Repository structure

```text
clinical-genomic-rag-eval-pro/
├─ .github/workflows/ci.yml
├─ configs/
│  ├─ default.yaml                 # pipeline/metrics/paths
│  └─ retrievers.yaml              # bm25 / dense / hybrid presets
├─ data/
│  ├─ corpus/
│  │  ├─ clin_note_001.md
│  │  ├─ genomics_oncokb_excerpt.md
│  │  └─ pubmed_abstract_001.md
│  ├─ eval_questions.json
│  └─ schemas/
│     ├─ answer_template.json
│     └─ doc_schema.md
├─ reports/
│  └─ .gitkeep                     # artifacts land here (jsonl/csv/html)
├─ src/
│  ├─ __init__.py
│  ├─ chunking.py                  # section-aware chunking (HPI / A&P / Pathology)
│  ├─ retriever_bm25.py            # BM25-lite retriever (pure Python)
│  ├─ retriever_dense.py           # Sentence-Transformers + FAISS (optional)
│  ├─ retriever_hybrid.py          # linear fusion α*BM25 + (1-α)*dense
│  ├─ llm_client.py                # answer stub + interface for real LLMs
│  ├─ pipeline.py                  # retrieve → cite → answer stub
│  ├─ eval_metrics.py              # hit@k, citation_recall, coverage, overlap, faithfulness_stub
│  ├─ faithfulness.py              # claim/evidence token Precision, Recall, F1
│  ├─ error_taxonomy.py            # rule-based error tags
│  ├─ reporting/
│  │  ├─ report_html.py            # JSONL → HTML summary
│  │  └─ plots.py                  # placeholder for charts
│  ├─ runners/
│  │  ├─ run_eval.py               # main evaluation runner
│  │  ├─ run_retrieval_bench.py    # retrieval-only benchmark (BM25 baseline)
│  │  └─ run_llm_ablate.py         # (stub) A/B harness for LLMs
│  └─ utils/
│     ├─ io.py                     # IO helpers (JSON/JSONL, dirs)
│     └─ seed.py                   # determinism
├─ tests/
│  ├─ test_metrics.py
│  ├─ test_faithfulness.py
│  ├─ test_retrieval_bench.py
│  └─ test_runner_smoke.py
├─ CHANGELOG.md
├─ LICENSE
├─ Makefile
├─ README.md
├─ requirements.txt                # base (pytest/pyyaml/pydantic)
├─ requirements-emb.txt            # + sentence-transformers, faiss-cpu
├─ requirements-llm.txt            # + openai (or another client)
├─ run.sh
└─ .gitignore



Installation

Base (BM25, reports, tests):

pip install -r requirements.txt


Dense / Hybrid retrieval (embeddings):

pip install -r requirements-emb.txt


LLM client (optional):

pip install -r requirements-llm.txt



Quick start

Run tests:

pytest -q



Evaluate (BM25 by default), build HTML:

python -m src.runners.run_eval --config configs/default.yaml
python -m src.reporting.report_html --in_jsonl reports/eval_report.jsonl --out_html reports/report.html


Or via Makefile:

make setup
make test
make eval
make report



Artifacts → reports/:

eval_report.jsonl — per-question records (answer, citations, metrics, tags)

eval_report.csv — per-question metric table

report.html — summary dashboard (averages + per-question table)


Retrieval modes

BM25 (default; no extra deps)
retrieval.kind: bm25 in configs/default.yaml.

Dense (requires requirements-emb.txt)
retrieval.kind: dense and choose model_name (e.g. sentence-transformers/all-MiniLM-L6-v2).

Hybrid (linear fusion)
retrieval.kind: hybrid, set hybrid_alpha ∈ [0,1] — weight for BM25.
Presets live in configs/retrievers.yaml.


Metrics (transparent proxies)

Retrieval: hit@k (gold doc present among top-k).

Citations: citation_recall (gold doc_ids referenced by the answer).

Grounding:

keyword_coverage (expected keywords found in claim).

context_overlap (token overlap between claim and concatenated citation quotes).

Faithfulness:

faithfulness_stub (every citation shares tokens with claim).

Faithfulness+: faithfulness_precision, faithfulness_recall, faithfulness_f1 — token-level alignment between claim and evidence quotes.

Error tags: simple rules (no_hit_at_k, no_citations, low_coverage, low_overlap) for quick triage.

These are auditable, deterministic proxies designed to reduce hallucinations and diagnose retrieval/prompt issues before integrating real clinical corpora.



Configuration (YAML)

configs/default.yaml:

paths:
  data_dir: data
  corpus_dir: data/corpus
  questions: data/eval_questions.json
  out_dir: reports

retrieval:
  kind: bm25          # bm25 | dense | hybrid
  top_k: 5
  hybrid_alpha: 0.5   # weight for BM25 in hybrid
  # model_name: sentence-transformers/all-MiniLM-L6-v2  # for dense/hybrid

generation:
  use_llm: false      # true -> src/llm_client.py
  max_chars: 280

metrics:
  hit_k: 5
  use_faithfulness_plus: true
  thresholds:
    low_coverage: 0.4
    low_overlap: 0.5



Console output example

=== Pro RAG evaluation complete ===
JSONL: /.../reports/eval_report.jsonl
CSV:   /.../reports/eval_report.csv
HTML:  /.../reports/report.html


Excerpt (CSV):

id,hit@k,citation_recall,keyword_coverage,context_overlap,faithfulness_stub,faithfulness_precision,faithfulness_recall,faithfulness_f1
q1,1.0,1.0,0.50,0.62,1.00,0.67,0.53,0.59
q2,1.0,1.0,1.00,0.71,1.00,0.72,0.60,0.65


Safety & scope

Corpus is synthetic and de-identified; intended for demo/evaluation only.

Metrics are transparent proxies; adapt/extend for production (guidelines, section weighting, claim–evidence alignment, human review).


Roadmap

Dense retriever presets + hybrid calibration scripts.

Stronger claim–evidence alignment and per-citation scoring.

Section weighting & guideline-aware boosts (e.g., NCCN/ESMO).

Per-question HTML dashboard with filters and plots.

LLM client adapter with auditable logs and A/B runner.


Versioning

We follow Semantic Versioning. See CHANGELOG.md
.
Initial release: v0.1.0-pro.



License

MIT — see LICENSE
.


Acknowledgments

This project draws on standard IR/RAG evaluation patterns and emphasizes reproducibility, auditability, and safety in clinical/genomic contexts.

