from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Set

from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer

LOGGER = logging.getLogger(__name__)


def load_words(word_list_path: Path) -> Set[str]:
    """Load allowed words from a newline-delimited file."""
    LOGGER.debug("Loading words from %s", word_list_path)
    with word_list_path.open("r", encoding="utf-8") as infile:
        return {line.strip().lower() for line in infile if line.strip()}


def tokenize(text: str, allowed_words: Set[str], stop_words: Set[str]) -> List[str]:
    """Tokenize text, filtering stop words and words not in the allowed list."""
    tokens = word_tokenize(text)
    return [
        token.lower()
        for token in tokens
        if token.lower() in allowed_words and token.lower() not in stop_words and len(token) >= 3
    ]


def wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def clean_directory(stories_dir: Path, word_list_path: Path) -> List[dict]:
    """Clean all .txt files in the provided directory."""
    stop_words = set(stopwords.words("english"))
    allowed_words = load_words(word_list_path) - stop_words

    documents = []
    lemma = WordNetLemmatizer()
    LOGGER.info("Cleaning stories from %s", stories_dir)

    for filename in sorted(stories_dir.iterdir()):
        if not filename.suffix == ".txt":
            continue
        with filename.open("r", encoding="utf-8", errors="ignore") as infile:
            doc_text = infile.read()

        tokens = tokenize(doc_text, allowed_words=allowed_words, stop_words=stop_words)
        tagged_tokens = pos_tag(tokens)
        stemmed_tokens = [
            lemma.lemmatize(word, wordnet_pos(tag)).lower() for word, tag in tagged_tokens
        ]
        documents.append({
            "filename": filename.name,
            "text": doc_text,
            "words": stemmed_tokens,
        })
    LOGGER.info("Cleaned %d stories", len(documents))
    return documents


def save_documents(documents: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing cleaned stories to %s", output_path)
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(list(documents), outfile, ensure_ascii=False)


def clean_and_save(
    stories_dir: Path,
    word_list_path: Path,
    output_path: Path,
) -> Path:
    documents = clean_directory(stories_dir=stories_dir, word_list_path=word_list_path)
    save_documents(documents, output_path=output_path)
    return output_path


def env_path(var_name: str, default: str) -> Path:
    return Path(os.getenv(var_name, default))
