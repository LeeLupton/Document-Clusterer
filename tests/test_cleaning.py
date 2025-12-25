from __future__ import annotations

from pathlib import Path
from typing import Iterable

import document_clusterer.cleaning as cleaning
from document_clusterer.cleaning import CleaningOptions, build_tokenizer, normalize_text
from document_clusterer.types import CleanedDocument
from pytest import MonkeyPatch


class DummyLemma:
    def lemmatize(self, word: str, pos: str) -> str:
        lowered = word.lower()
        if lowered.endswith("ing"):
            return lowered[:-3]
        if lowered.endswith("s"):
            return lowered[:-1]
        return lowered


def test_nltk_tokenization_and_lemmatization(monkeypatch: MonkeyPatch) -> None:
    sample_text = "The jumping DOGS quickly leap over trees."
    stop_words = {"the", "over"}
    options = CleaningOptions(min_token_length=3, pipeline="nltk")

    def fake_word_tokenize(text: str) -> list[str]:
        return [token.strip(".") for token in text.split()]

    def fake_pos_tag(tokens: Iterable[str]) -> list[tuple[str, str]]:
        tag_map = {
            "the": "DT",
            "jumping": "VBG",
            "dogs": "NNS",
            "quickly": "RB",
            "leap": "VB",
            "over": "IN",
            "trees": "NNS",
        }
        return [(token, tag_map[token]) for token in tokens]

    monkeypatch.setattr(cleaning, "word_tokenize", fake_word_tokenize)
    monkeypatch.setattr(cleaning, "pos_tag", fake_pos_tag)
    monkeypatch.setattr(cleaning, "WordNetLemmatizer", lambda: DummyLemma())
    monkeypatch.setattr(cleaning, "_wordnet_pos", lambda tag: "n")

    tokenizer = build_tokenizer(stop_words, allowed_words=None, options=options)
    normalized = normalize_text(sample_text, options)
    tokens = tokenizer(normalized)

    assert tokens == ["jump", "dog", "quickly", "leap", "tree"]


def test_clean_documents_returns_structured_output(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    sample_corpus = [("story1.txt", "Cats and cats play together.")]

    monkeypatch.setattr(cleaning, "word_tokenize", lambda text: text.split())
    monkeypatch.setattr(cleaning, "pos_tag", lambda tokens: [(token, "NN") for token in tokens])
    monkeypatch.setattr(cleaning, "WordNetLemmatizer", lambda: DummyLemma())
    monkeypatch.setattr(cleaning, "_wordnet_pos", lambda tag: "n")
    monkeypatch.setattr(cleaning, "load_stop_words", lambda *args, **kwargs: set())

    documents = cleaning.clean_documents(sample_corpus, options=CleaningOptions())
    expected: CleanedDocument = {
        "filename": "story1.txt",
        "text": "Cats and cats play together.",
        "words": ["cat", "and", "cat", "play", "together."],
    }

    assert documents == [expected]
