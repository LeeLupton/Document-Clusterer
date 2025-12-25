from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from document_clusterer import build_document_term_matrix, cluster_documents
from document_clusterer.types import CleanedDocument
from pytest import MonkeyPatch


def test_build_document_term_matrix() -> None:
    documents: list[CleanedDocument] = [
        {"filename": "a.txt", "text": "a", "words": ["dog", "runs", "dog"]},
        {"filename": "b.txt", "text": "b", "words": ["cat", "runs"]},
    ]

    matrix, vocabulary = build_document_term_matrix(documents)

    assert vocabulary == ["cat", "dog", "runs"]
    np.testing.assert_array_equal(
        matrix,
        np.array(
            [
                [0, 2, 1],
                [1, 0, 1],
            ]
        ),
    )


def test_cluster_documents_smoke(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    documents: list[CleanedDocument] = [
        {"filename": "story1.txt", "text": "", "words": ["fox", "jump"]},
        {"filename": "story2.txt", "text": "", "words": ["document", "tiny"]},
    ]
    data_path = tmp_path / "sample.json"
    data_path.write_text(json.dumps(documents), encoding="utf-8")

    fake_embeddings = np.array([[0.0], [10.0]])
    monkeypatch.setattr(
        "document_clusterer.model.embed_documents",
        lambda docs, model_name="": fake_embeddings,
    )

    output_dir = tmp_path / "output"
    clusters = cluster_documents(
        data_path=data_path,
        stories_dir=Path("data/sample"),
        output_dir=output_dir,
        model_name="dummy-model",
        cluster_method="kmeans",
        cluster_count=2,
        reduction_method=None,
        summary_top_n=2,
        assignments_basename="assignments",
    )

    assert (output_dir / "assignments.json").exists()
    assert (output_dir / "assignments.csv").exists()
    assert (output_dir / "cluster_summaries.json").exists()
    assert set(clusters.keys()) == {"0", "1"}
    assert {fname for names in clusters.values() for fname in names} == {"story1.txt", "story2.txt"}
