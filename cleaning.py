"""Compatibility wrapper for cleaning CNN stories.

Use the packaged CLI instead:
    python -m document_clusterer.cli clean
or install the package and run `document-clusterer clean`.
"""

from document_clusterer.cli import clean_cli


if __name__ == "__main__":
    clean_cli()
