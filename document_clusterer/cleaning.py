from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Set

from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer

from document_clusterer.types import CleanedDocument

LOGGER = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)*\b")


@dataclass(frozen=True)
class CleaningOptions:
    """Configuration for text cleaning."""

    lowercase: bool = True
    strip_urls: bool = True
    strip_numbers: bool = True
    pipeline: str = "nltk"  # "nltk" or "spacy"
    spacy_model: str = "en_core_web_sm"
    min_token_length: int = 3
    include_default_stopwords: bool = True


def env_path(var_name: str, default: str) -> Path:
    return Path(os.getenv(var_name, default))


def load_corpus(stories_dir: Path) -> list[tuple[str, str]]:
    """Load raw text corpus from a directory of .txt files."""

    LOGGER.info("Loading corpus from %s", stories_dir)
    corpus: list[tuple[str, str]] = []
    for filename in sorted(stories_dir.iterdir()):
        if filename.suffix != ".txt":
            continue
        with filename.open("r", encoding="utf-8", errors="ignore") as infile:
            corpus.append((filename.name, infile.read()))
    LOGGER.debug("Loaded %d documents", len(corpus))
    return corpus


def load_words(word_list_path: Path | None) -> Set[str] | None:
    """Load allowed words from a newline-delimited file."""

    if word_list_path is None:
        return None

    LOGGER.debug("Loading words from %s", word_list_path)
    with word_list_path.open("r", encoding="utf-8") as infile:
        return {line.strip().lower() for line in infile if line.strip()}


def load_stop_words(
    stop_words_path: Path | None,
    *,
    include_default: bool = True,
    extra_stopwords: Sequence[str] | None = None,
    lowercase: bool = True,
) -> Set[str]:
    """Compose a stop word list using optional defaults, file, and extras."""

    compiled: Set[str] = set()

    if include_default:
        compiled.update(stopwords.words("english"))

    if stop_words_path:
        LOGGER.debug("Loading stop words from %s", stop_words_path)
        with stop_words_path.open("r", encoding="utf-8") as infile:
            compiled.update(line.strip() for line in infile if line.strip())

    if extra_stopwords:
        compiled.update(extra_stopwords)

    if lowercase:
        compiled = {word.lower() for word in compiled}

    return compiled


def normalize_text(text: str, options: CleaningOptions) -> str:
    cleaned = text
    if options.strip_urls:
        cleaned = URL_PATTERN.sub(" ", cleaned)
    if options.strip_numbers:
        cleaned = NUMBER_PATTERN.sub(" ", cleaned)
    if options.lowercase:
        cleaned = cleaned.lower()
    return cleaned


def _nltk_tokenizer(
    stop_words: Set[str], allowed_words: Set[str] | None, options: CleaningOptions
) -> Callable[[str], List[str]]:
    lemma = WordNetLemmatizer()

    def tokenizer(text: str) -> List[str]:
        tokens = word_tokenize(text)
        filtered = [
            token
            for token in tokens
            if len(token) >= options.min_token_length
            and (allowed_words is None or token.lower() in allowed_words)
            and token.lower() not in stop_words
        ]
        tagged_tokens = pos_tag(filtered)
        return [lemma.lemmatize(word, _wordnet_pos(tag)).lower() for word, tag in tagged_tokens]

    return tokenizer


def _wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return str(wordnet.ADJ)
    if tag.startswith("V"):
        return str(wordnet.VERB)
    if tag.startswith("N"):
        return str(wordnet.NOUN)
    if tag.startswith("R"):
        return str(wordnet.ADV)
    return str(wordnet.NOUN)


def _load_spacy_model(model_name: str) -> Any:
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("spaCy is not installed. Install it to use the spaCy pipeline.") from exc

    try:
        return spacy.load(model_name)
    except OSError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. Run 'python -m spacy download {model_name}'."
        ) from exc


def _spacy_tokenizer(
    stop_words: Set[str], allowed_words: Set[str] | None, options: CleaningOptions
) -> Callable[[str], List[str]]:
    nlp = _load_spacy_model(options.spacy_model)

    def tokenizer(text: str) -> List[str]:
        doc = nlp(text)
        tokens: List[str] = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            lemma = token.lemma_ if token.lemma_ != "-PRON-" else token.text
            normalized = lemma.lower() if options.lowercase else lemma
            if len(normalized) < options.min_token_length:
                continue
            if allowed_words is not None and normalized.lower() not in allowed_words:
                continue
            if normalized.lower() in stop_words or token.is_stop:
                continue
            tokens.append(normalized.lower())
        return tokens

    return tokenizer


def build_tokenizer(
    stop_words: Set[str],
    allowed_words: Set[str] | None,
    options: CleaningOptions,
) -> Callable[[str], List[str]]:
    if options.pipeline == "nltk":
        return _nltk_tokenizer(stop_words, allowed_words, options)
    if options.pipeline == "spacy":
        return _spacy_tokenizer(stop_words, allowed_words, options)
    raise ValueError(f"Unknown pipeline '{options.pipeline}'. Expected 'nltk' or 'spacy'.")


def clean_documents(
    corpus: Iterable[tuple[str, str]],
    *,
    word_list_path: Path | None = None,
    stop_words_path: Path | None = None,
    extra_stopwords: Sequence[str] | None = None,
    options: CleaningOptions | None = None,
) -> List[CleanedDocument]:
    """Clean a pre-loaded corpus and return structured documents."""

    cleaning_options = options or CleaningOptions()
    allowed_words = load_words(word_list_path)
    stop_words = load_stop_words(
        stop_words_path,
        include_default=cleaning_options.include_default_stopwords,
        extra_stopwords=extra_stopwords,
        lowercase=cleaning_options.lowercase,
    )
    tokenizer = build_tokenizer(stop_words, allowed_words, cleaning_options)

    documents: List[CleanedDocument] = []
    for filename, text in corpus:
        normalized_text = normalize_text(text, cleaning_options)
        tokens = tokenizer(normalized_text)
        documents.append(
            {
                "filename": filename,
                "text": text,
                "words": tokens,
            }
        )

    LOGGER.info("Cleaned %d documents", len(documents))
    return documents


def write_json(documents: Iterable[CleanedDocument], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing cleaned stories to %s", output_path)
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(list(documents), outfile, ensure_ascii=False)


def clean_directory(
    stories_dir: Path,
    word_list_path: Path | None,
    *,
    stop_words_path: Path | None = None,
    extra_stopwords: Sequence[str] | None = None,
    options: CleaningOptions | None = None,
) -> List[CleanedDocument]:
    """Clean all .txt files in the provided directory."""

    corpus = load_corpus(stories_dir)
    return clean_documents(
        corpus,
        word_list_path=word_list_path,
        stop_words_path=stop_words_path,
        extra_stopwords=extra_stopwords,
        options=options,
    )


def save_documents(documents: Iterable[CleanedDocument], output_path: Path) -> None:
    write_json(documents, output_path)


def clean_and_save(
    stories_dir: Path,
    word_list_path: Path,
    output_path: Path,
    *,
    stop_words_path: Path | None = None,
    extra_stopwords: Sequence[str] | None = None,
    options: CleaningOptions | None = None,
) -> Path:
    documents = clean_directory(
        stories_dir=stories_dir,
        word_list_path=word_list_path,
        stop_words_path=stop_words_path,
        extra_stopwords=extra_stopwords,
        options=options,
    )
    write_json(documents, output_path=output_path)
    return output_path
