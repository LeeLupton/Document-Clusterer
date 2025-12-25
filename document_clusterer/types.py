from __future__ import annotations

from typing import TypedDict


class CleanedDocument(TypedDict):
    """Structured document produced by the cleaning pipeline."""

    filename: str
    text: str
    words: list[str]
