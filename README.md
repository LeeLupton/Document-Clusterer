# Document-Clusterer

A simple document clustering utility using singular value decomposition (SVD) on a corpus of CNN stories.

## Installation

```bash
pip install -r requirements.txt
```

You can also install the package in editable mode:

```bash
pip install -e .
```

## Usage

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
document-clusterer cluster --input-file all_stories.json --stories-dir data/cnn-stories --output-dir clusteredDocuments --clusters 10 --rank 10
```

Environment variable defaults:

* `INPUT_JSON` – cleaned JSON input (default: `all_stories.json`)
* `STORIES_DIR` – directory containing original stories (default: `data/cnn-stories`)
* `CLUSTER_OUTPUT_DIR` – output directory for clustered documents (default: `clusteredDocuments`)
* `CLUSTER_COUNT` – number of clusters to create (default: `10`)
* `SVD_RANK` – rank used for the SVD approximation (default: `10`)

## Development

Both legacy scripts remain as compatibility wrappers:

* `cleaning.py` now delegates to the packaged cleaner
* `model.py` delegates to the packaged clustering CLI

Feel free to extend `document_clusterer/` to add additional cleaning or clustering options.
