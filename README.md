# Document-Clusterer

A simple document clustering utility that cleans text, builds SentenceTransformers embeddings, and clusters with KMeans or HDBSCAN.

## Setup

Requirements: Python 3.10+.

1) Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

2) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3) Download the NLTK data used by the cleaner (one-time):

```bash
python - <<'PY'
import nltk

for resource in [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "stopwords",
    "wordnet",
    "omw-1.4",
]:
    nltk.download(resource)
PY
```

4) (Optional) Extras for the notebook and visualizations:

```bash
pip install pandas matplotlib seaborn plotly
```

## Installation

```bash
pip install -r requirements.txt
```

You can also install the package in editable mode:

```bash
pip install -e .
```

## Usage

## Quickstart

### Clean stories

```bash
document-clusterer clean \
  --stories-dir data/cnn-stories \
  --word-list data/one-grams.txt \
  --output all_stories.json \
  --pipeline nltk \
  --min-token-length 3
```

Environment variable defaults are available:

* `STORIES_DIR` – directory containing story `.txt` files (default: `data/cnn-stories`)
* `WORD_LIST_PATH` – newline-delimited allowed words (default: `data/one-grams.txt`)
* `STOP_WORDS_PATH` – optional newline-delimited stopword list to merge or override defaults
* `OUTPUT_JSON` – JSON output path (default: `all_stories.json`)
* `CLEANING_PIPELINE` – tokenizer pipeline (`nltk` or `spacy`, default: `nltk`)
* `SPACY_MODEL` – spaCy model name (default: `en_core_web_sm`)
* `MIN_TOKEN_LENGTH` – minimum token length to retain (default: `3`)

Switch pipelines with `--pipeline spacy` (requires spaCy and a downloaded model). Toggle lowercasing, URL stripping, and number stripping via `--no-lowercase`, `--keep-urls`, and `--keep-numbers`.

For a quick demo, a tiny sample corpus is available in `data/sample`:

```bash
document-clusterer clean --stories-dir data/sample --output data/sample.json
```

### Cluster cleaned output

```bash
document-clusterer cluster \
  --input-file all_stories.json \
  --stories-dir data/cnn-stories \
  --output-dir clusteredDocuments \
  --model-name all-MiniLM-L6-v2 \
  --cluster-method kmeans \
  --clusters 10 \
  --reduction umap \
  --reduction-dim 2
```

On Windows, you can run the same commands in PowerShell. If `document-clusterer` is not on your `PATH`, use the module form instead:

```powershell
# From the repo root
python -m document_clusterer.cli clean --stories-dir data\sample --output data\sample_cleaned.json
python -m document_clusterer.cli cluster --input-file data\sample_cleaned.json --stories-dir data\sample --output-dir clusteredDocuments\sample
```

Common Windows tips:

* Run commands from the repository root so relative paths like `data\sample_cleaned.json` resolve correctly.
* Prefer backslashes in PowerShell (`data\sample`) and ensure the cleaned JSON exists (run the clean step before clustering).

python -m document_clusterer.cli clean --stories-dir data\\sample --output data\\sample.json
python -m document_clusterer.cli cluster --input-file data\\sample_cleaned.json --stories-dir data\\sample --output-dir clusteredDocuments\\sample
```

Environment variable defaults:

* `INPUT_JSON` – cleaned JSON input (default: `all_stories.json`)
* `STORIES_DIR` – directory containing original stories (default: `data/cnn-stories`)
* `CLUSTER_OUTPUT_DIR` – output directory for clustered documents (default: `clusteredDocuments`)
* `CLUSTER_COUNT` – number of clusters for KMeans (default: `10`)
* `CLUSTER_METHOD` – `kmeans` or `hdbscan` (default: `kmeans`)
* `EMBEDDING_MODEL` – SentenceTransformers model name (default: `all-MiniLM-L6-v2`)
* `KMEANS_RANDOM_STATE` – seed for KMeans initialization (default: `42`)
* `HDBSCAN_MIN_CLUSTER_SIZE` / `HDBSCAN_MIN_SAMPLES` – HDBSCAN hyperparameters
* `REDUCTION_METHOD` – `umap`, `pca`, or `none` (default: `umap`)
* `REDUCTION_DIM`, `UMAP_NEIGHBORS`, `UMAP_MIN_DIST` – dimensionality reduction controls
* `SUMMARY_TERMS` – number of top terms per cluster in summaries (default: `10`)
* `ASSIGNMENTS_BASENAME` – base filename for assignment JSON/CSV outputs (default: `cluster_assignments`)

Outputs now include:

* `cluster_assignments.json` and `.csv` – cluster labels (and visualization coordinates if enabled)
* `cluster_summaries.json` / `.txt` – human-friendly top terms per cluster
* Subdirectories under `clusteredDocuments/` for each cluster label (with `noise` for HDBSCAN outliers)

### End-to-end example (sample data)

```bash
# Clean a lightweight demo corpus
document-clusterer clean \
  --stories-dir data/sample \
  --word-list data/one-grams.txt \
  --output data/sample_cleaned.json

# Cluster with KMeans + UMAP for 2D visualization
document-clusterer cluster \
  --input-file data/sample_cleaned.json \
  --stories-dir data/sample \
  --output-dir clusteredDocuments/sample \
  --model-name all-MiniLM-L6-v2 \
  --cluster-method kmeans \
  --clusters 3 \
  --reduction umap \
  --reduction-dim 2
```

After clustering, check `clusteredDocuments/sample/cluster_assignments.json` for labels and UMAP coordinates and `cluster_summaries.json` for the top keywords per cluster.

## Development

Both legacy scripts remain as compatibility wrappers:

* `cleaning.py` now delegates to the packaged cleaner
* `model.py` delegates to the packaged clustering CLI

Feel free to extend `document_clusterer/` to add additional cleaning or clustering options.

## Architecture overview

* **CLI entrypoints** – `document_clusterer/cli.py` exposes `clean` and `cluster` subcommands plus convenience wrappers (`document-clusterer-clean`, `document-clusterer-cluster`).
* **Cleaning pipeline** – `document_clusterer/cleaning.py` loads `.txt` files, normalizes text (URL/number stripping, lowercasing), tokenizes with NLTK or spaCy, removes stop words/short tokens, lemmatizes, and writes structured JSON via `save_documents`.
* **Embedding + clustering** – `document_clusterer/model.py` encodes documents with SentenceTransformers (`embed_documents`), clusters with KMeans or HDBSCAN (`run_clustering`), optionally reduces dimensions with UMAP/PCA for visualization (`reduce_embeddings`), and summarizes top terms per cluster (`summarize_clusters`).
* **Outputs** – assignments and summaries are persisted as JSON/CSV/text, and the original `.txt` files are copied into per-cluster folders for inspection.

## Notebook walkthrough

A portfolio-ready Jupyter notebook lives in `notebooks/Document_Clusterer_Demo.ipynb`. It demonstrates:

1. Building a small themed corpus on the fly (tech, sports, health).
2. Cleaning the text with `CleaningOptions` and viewing tokenized results.
3. Running KMeans + UMAP via `cluster_documents` to produce assignments and summaries.
4. Visualizing clusters in 2D and charting top keywords per cluster with inline matplotlib outputs.

Launch with:

```bash
jupyter notebook notebooks/Document_Clusterer_Demo.ipynb
```
