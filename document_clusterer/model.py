from __future__ import annotations

import csv
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

LOGGER = logging.getLogger(__name__)


def env_path(var_name: str, default: str) -> Path:
    return Path(os.getenv(var_name, default))


def env_int(var_name: str, default: int) -> int:
    return int(os.getenv(var_name, str(default)))


def load_documents(data_path: Path) -> List[Dict]:
    LOGGER.info("Loading documents from %s", data_path)
    with data_path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def embed_documents(documents: Sequence[Dict], model_name: str) -> np.ndarray:
    LOGGER.info("Encoding %d documents with SentenceTransformer model '%s'", len(documents), model_name)
    model = SentenceTransformer(model_name)
    texts = []
    for document in documents:
        if document.get("text"):
            texts.append(document["text"])
        else:
            texts.append(" ".join(document.get("words", [])))
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.asarray(embeddings)


def run_clustering(
    embeddings: np.ndarray,
    *,
    cluster_method: str,
    cluster_count: int | None,
    kmeans_random_state: int | None,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int | None,
) -> np.ndarray:
    if cluster_method == "kmeans":
        if cluster_count is None:
            raise ValueError("cluster_count must be provided when using kmeans clustering.")
        if cluster_count < 1:
            raise ValueError("cluster_count must be at least 1.")
        LOGGER.info("Running KMeans with k=%d", cluster_count)
        model = KMeans(n_clusters=cluster_count, random_state=kmeans_random_state, n_init="auto")
        return model.fit_predict(embeddings)

    if cluster_method == "hdbscan":
        try:
            import hdbscan
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("hdbscan is not installed. Install it to use HDBSCAN clustering.") from exc

        LOGGER.info(
            "Running HDBSCAN with min_cluster_size=%d%s",
            hdbscan_min_cluster_size,
            f", min_samples={hdbscan_min_samples}" if hdbscan_min_samples is not None else "",
        )
        model = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric="euclidean",
        )
        return model.fit_predict(embeddings)

    raise ValueError(f"Unknown cluster_method '{cluster_method}'. Expected 'kmeans' or 'hdbscan'.")


def reduce_embeddings(
    embeddings: np.ndarray,
    *,
    reduction_method: str | None,
    reduction_dim: int,
    umap_neighbors: int,
    umap_min_dist: float,
    random_state: int | None,
) -> np.ndarray | None:
    if reduction_method is None or reduction_method == "none":
        return None

    if reduction_method == "pca":
        LOGGER.info("Reducing embeddings with PCA to %d dimensions", reduction_dim)
        reducer = PCA(n_components=reduction_dim, random_state=random_state)
        return reducer.fit_transform(embeddings)

    if reduction_method == "umap":
        try:
            import umap  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            import umap.umap_ as umap  # type: ignore
        LOGGER.info(
            "Reducing embeddings with UMAP to %d dimensions (n_neighbors=%d, min_dist=%.2f)",
            reduction_dim,
            umap_neighbors,
            umap_min_dist,
        )
        reducer = umap.UMAP(
            n_components=reduction_dim,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            random_state=random_state,
        )
        return reducer.fit_transform(embeddings)

    raise ValueError(f"Unknown reduction_method '{reduction_method}'. Expected 'umap', 'pca', or 'none'.")


def summarize_clusters(documents: Sequence[Dict], labels: np.ndarray, top_n: int) -> Dict[int, list[tuple[str, int]]]:
    clustered_terms: Dict[int, Counter] = defaultdict(Counter)
    for document, label in zip(documents, labels):
        clustered_terms[int(label)].update(document.get("words", []))

    summaries: Dict[int, list[tuple[str, int]]] = {}
    for label, counter in clustered_terms.items():
        summaries[label] = counter.most_common(top_n)
    return summaries


def save_assignments(
    documents: Sequence[Dict],
    labels: np.ndarray,
    output_dir: Path,
    *,
    basename: str,
    reduced_embeddings: np.ndarray | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{basename}.json"
    csv_path = output_dir / f"{basename}.csv"

    records: list[dict] = []
    for idx, (document, label) in enumerate(zip(documents, labels)):
        record: dict = {
            "id": idx,
            "filename": document["filename"],
            "cluster": int(label),
        }
        if reduced_embeddings is not None:
            for dim_index, value in enumerate(reduced_embeddings[idx]):
                record[f"dim_{dim_index}"] = float(value)
        records.append(record)

    LOGGER.info("Writing cluster assignments to %s and %s", json_path, csv_path)
    with json_path.open("w", encoding="utf-8") as outfile:
        json.dump(records, outfile, ensure_ascii=False, indent=2)

    fieldnames = list(records[0].keys()) if records else ["id", "filename", "cluster"]
    with csv_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return json_path


def save_summaries(summaries: Dict[int, list[tuple[str, int]]], output_dir: Path) -> None:
    json_path = output_dir / "cluster_summaries.json"
    text_path = output_dir / "cluster_summaries.txt"

    LOGGER.info("Writing cluster summaries to %s and %s", json_path, text_path)
    with json_path.open("w", encoding="utf-8") as outfile:
        json.dump(
            {str(label): [{"term": term, "count": count} for term, count in terms] for label, terms in summaries.items()},
            outfile,
            ensure_ascii=False,
            indent=2,
        )

    with text_path.open("w", encoding="utf-8") as outfile:
        for label, terms in sorted(summaries.items(), key=lambda item: item[0]):
            outfile.write(f"Cluster {label}:\n")
            if not terms:
                outfile.write("  (no terms)\n\n")
                continue
            for term, count in terms:
                outfile.write(f"  {term}: {count}\n")
            outfile.write("\n")


def copy_clusters(document_clusters: Dict[str, List[str]], stories_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for cluster_label, clustered_documents in document_clusters.items():
        destination = output_dir / cluster_label
        destination.mkdir(parents=True, exist_ok=True)
        for clustered_document_name in clustered_documents:
            srcfile = stories_dir / clustered_document_name
            dstfile = destination / clustered_document_name
            LOGGER.debug("Copying %s -> %s", srcfile, dstfile)
            dstfile.parent.mkdir(parents=True, exist_ok=True)
            copy_if_exists(srcfile, dstfile)


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        LOGGER.warning("Source file %s does not exist; skipping copy.", src)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


def cluster_documents(
    data_path: Path,
    stories_dir: Path,
    output_dir: Path,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    cluster_method: str = "kmeans",
    cluster_count: int | None = 10,
    kmeans_random_state: int | None = 42,
    hdbscan_min_cluster_size: int = 5,
    hdbscan_min_samples: int | None = None,
    reduction_method: str | None = "umap",
    reduction_dim: int = 2,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    reduction_random_state: int | None = 42,
    summary_top_n: int = 10,
    assignments_basename: str = "cluster_assignments",
) -> Dict[str, List[str]]:
    documents = load_documents(data_path)
    if not documents:
        raise ValueError("No documents found to cluster.")

    embeddings = embed_documents(documents, model_name=model_name)
    labels = run_clustering(
        embeddings,
        cluster_method=cluster_method,
        cluster_count=cluster_count,
        kmeans_random_state=kmeans_random_state,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
    )

    reduced_embeddings = reduce_embeddings(
        embeddings,
        reduction_method=reduction_method,
        reduction_dim=reduction_dim,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        random_state=reduction_random_state,
    )

    save_assignments(
        documents,
        labels,
        output_dir=output_dir,
        basename=assignments_basename,
        reduced_embeddings=reduced_embeddings,
    )

    summaries = summarize_clusters(documents, labels, top_n=summary_top_n)
    save_summaries(summaries, output_dir)

    cluster_to_documents: Dict[str, List[str]] = defaultdict(list)
    for document, label in zip(documents, labels):
        cluster_label = str(label) if label != -1 else "noise"
        cluster_to_documents[cluster_label].append(document["filename"])

    copy_clusters(cluster_to_documents, stories_dir, output_dir)
    return cluster_to_documents

