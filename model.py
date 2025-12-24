"""Compatibility wrapper for clustering cleaned CNN stories.

Use the packaged CLI instead:
    python -m document_clusterer.cli cluster
or install the package and run `document-clusterer cluster`.
"""

from document_clusterer.cli import cluster_cli


if __name__ == "__main__":
    cluster_cli()
