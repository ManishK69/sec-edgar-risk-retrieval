# SEC EDGAR Risk Intelligence Retrieval System

**CAP 6776 — Information Retrieval | Spring 2026**

A multi-model Information Retrieval System over SEC EDGAR 10-K filings, letting
risk analysts, compliance officers, and researchers query risk disclosures
(e.g. Item 1A Risk Factors) in natural language — something EDGAR's native
keyword search cannot do.

## Approach

191 10-K filings from SEC EDGAR were segmented by section into **2,807 document
chunks** (~1,200 words each). Three retrieval methods are compared:

| Method | Description |
| ------ | ----------- |
| **BM25** | Sparse keyword retrieval (`rank_bm25`, k1=1.5, b=0.75) |
| **Dense bi-encoder** | Semantic search with `bge-base-en-v1.5` (selected empirically over FinBERT) |
| **Retrieve & re-rank** | BGE top-50 candidates rescored with `cross-encoder/ms-marco-MiniLM-L-12-v2` |

A Gradio web UI shows all three methods side by side (rank, doc ID, relevance,
snippet) with a Top-K slider.

## Results

Evaluated with pool-based human relevance judgments — 5 queries, 118 pooled
documents, graded 0/1/2 — at K=10:

| Metric | BM25 | Dense (BGE) | Re-rank (Cross-enc.) |
| ------ | ---- | ----------- | -------------------- |
| P@10   | 0.96 | 0.80        | 0.78                 |
| R@10   | 0.55 | 0.37        | 0.35                 |
| MAP    | 0.953| 0.776       | 0.767                |

BM25 leads because SEC filings use a controlled regulatory vocabulary where
exact match shines. Empirical encoder selection (FinBERT → BGE) lifted dense
MAP by 30% (0.597 → 0.776). The cross-encoder adds little once Stage-1
retrieval is already strong — and both neural methods fail on ambiguous terms
like "environment"/"climate", a concrete finding about general embeddings on
financial text. Full analysis in `paper/Final_Project_Memo.pdf`.

## Structure

```
├── code/       # main.ipynb (full pipeline), requirements.txt
├── data/       # corpus.csv, queries.csv, qrels.csv (+ pooled qrels)
├── output/     # metric plots and summaries
└── paper/      # Final_Project_Memo.pdf (executive memo)
```

## Setup

See `code/README.md` for full instructions. Quick start (local):

```bash
pip install -r code/requirements.txt
python -m nltk.downloader punkt stopwords wordnet punkt_tab
jupyter notebook code/main.ipynb
```

Run the notebook cells top to bottom: corpus processing → BM25 → dense
embeddings → cross-encoder re-ranking → evaluation → Gradio UI.
