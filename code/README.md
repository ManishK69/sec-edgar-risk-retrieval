# SEC EDGAR Information Retrieval System
### CAP 6776 — Information Retrieval | Spring 2026

## Overview

This project implements a multi-model Information Retrieval System (IRS) over a corpus of SEC EDGAR 10-K filings. Given a natural language query, the system retrieves and ranks the most relevant document sections (e.g., Item 1A Risk Factors) using three retrieval methods:

1. **BM25** — Traditional sparse keyword retrieval
2. **Dense Bi-Encoder** — Semantic search using sentence embeddings
3. **Retrieve & Re-Rank** — Two-stage pipeline using a Cross-Encoder

A Gradio web UI allows side-by-side comparison of all three methods for any query.


## Project Structure

Project_LNU_Manish/
├── code/
│   ├── main.ipynb       # Main notebook (all cells)
│   ├── README.md             # This file
│   └── requirements.txt      # Python dependencies
├── data/
│   ├── corpus.csv            # Processed SEC EDGAR document chunks
│   ├── queries.csv           # 5 evaluation queries
│   └── qrels.csv             # Relevance judgments (manually reviewed)
├── outputs/
│   ├── metrics_summary.csv         # Per-query evaluation metrics
│   ├── mean_metrics_comparison.png # Grouped bar chart
│   └── per_query_ap_comparison.png # Per-query AP line plot




## Setup Instructions

### Requirements

- Python 3.9+
- GPU recommended (tested on Google Colab A100)
- Google Drive access (corpus JSON files must be uploaded to Drive)

### Option 1: Google Colab (Recommended)

1. Upload `Untitled2.ipynb` to Google Colab.
2. Upload your SEC EDGAR JSON files to Google Drive under a folder (e.g., `MyDrive/extracted/`).
3. Install dependencies — each cell that requires a package runs `pip install` automatically.
4. Run cells top to bottom. Update the `drive_path` variable in Cell 0 to match your Drive folder path:
   ```python
   drive_path = "/content/drive/MyDrive/extracted"
   ```

### Option 2: Local Environment

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet punkt_tab
jupyter notebook Untitled2.ipynb
```

## Running the System

Run all notebook cells in order:

| Cell | Description |
|------|-------------|
| 0 | Mount Drive, process JSON files → `corpus.csv` |
| 1 | Text preprocessing (tokenization, stopword removal, lemmatization) |
| 2–3 | BM25 index + search function |
| 4–5 | Dense bi-encoder embeddings + search function |
| 6 | Cross-encoder re-ranker + two-stage search function |
| 7 | Evaluation (P@10, R@10, MAP) across all three methods |
| 8 | Visualization (bar chart + per-query line plot) |
| 9–10 | Gradio UI — launches interactive search interface |

---

## Evaluation

- **5 queries** with relevance judgments (`qrels.csv`)
- **Metrics computed:** P@10, R@10, MAP
- Relevance judgments were generated using a cross-encoder judge model and manually reviewed for accuracy

---

## AI Tool Disclosure

Used Claude for research, review and evaluation of code and memo.

## Dependencies

See `requirements.txt` for the full list.

## Video Walkthrough Link

https://drive.google.com/file/d/1s50GmTaZkxF56WMmilZy42s7dGIckMSh/view?usp=drive_link