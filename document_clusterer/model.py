from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from shutil import copyfile
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.linalg import svd
from scipy.cluster.vq import kmeans2

LOGGER = logging.getLogger(__name__)


def low_rank_approx(matrix: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a k-rank approximation of a matrix."""
    u, sigma, v = svd(matrix, full_matrices=False)

    ar = np.zeros((len(u), len(v)))
    max_rank = min(k, len(sigma))
    for i in range(max_rank):
        ar += sigma[i] * np.outer(u.T[i], v[i])
    return u[:, :k], ar, v[:k, :]


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize a document-term matrix."""
    num_words, num_docs = matrix.shape
    local_factors = np.log(np.ones(matrix.shape) + matrix.copy())

    probabilities = matrix.copy()
    row_sums = np.sum(matrix, axis=1)

    if not np.all(row_sums > 0):
        raise ValueError("All rows must have non-zero sums for normalization.")

    probabilities = (probabilities.T / row_sums).T

    entropies = (probabilities * np.ma.log(probabilities).filled(0) / np.log(num_docs))
    global_factors = np.ones(num_words) + np.sum(entropies, axis=1)

    normalized_matrix = (local_factors.T * global_factors).T
    return normalized_matrix


def make_document_term_matrix(data: Sequence[Dict]) -> Tuple[np.ndarray, Tuple[Dict[int, str], Dict[int, Dict]]]:
    words = all_words(data)
    word_to_index = {word: i for i, word in enumerate(words)}
    index_to_word = dict(enumerate(words))
    index_to_document = dict(enumerate(data))

    matrix = np.zeros((len(words), len(data)))
    for doc_id, document in enumerate(data):
        doc_words = Counter(document["words"])
        for word, count in doc_words.items():
            matrix[word_to_index[word], doc_id] = count

    return matrix, (index_to_word, index_to_document)


def cluster(vectors: np.ndarray, cluster_count: int):
    """Run k-means clustering on document vectors."""
    return kmeans2(vectors, k=cluster_count, minit="++")


def all_words(data: Iterable[Dict]) -> List[str]:
    words = set()
    for entry in data:
        words |= set(entry["words"])
    return list(sorted(words))


def load_documents(data_path: Path) -> List[Dict]:
    LOGGER.info("Loading documents from %s", data_path)
    with data_path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def copy_clusters(
    document_clusters: List[List[str]],
    stories_dir: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for cluster_number, clustered_documents in enumerate(document_clusters, start=1):
        destination = output_dir / str(cluster_number)
        destination.mkdir(parents=True, exist_ok=True)
        for clustered_document_name in clustered_documents:
            srcfile = stories_dir / clustered_document_name
            dstfile = destination / clustered_document_name
            LOGGER.debug("Copying %s -> %s", srcfile, dstfile)
            copyfile(srcfile, dstfile)


def cluster_documents(
    data_path: Path,
    stories_dir: Path,
    output_dir: Path,
    cluster_count: int = 10,
    rank: int = 10,
) -> List[List[str]]:
    data = load_documents(data_path)
    if cluster_count < 1:
        raise ValueError("cluster_count must be at least 1.")
    if cluster_count > len(data):
        raise ValueError("cluster_count cannot exceed the number of documents.")

    matrix, (index_to_word, index_to_document) = make_document_term_matrix(data)
    matrix = normalize(matrix)
    LOGGER.info("Classifying the data into %d parts", cluster_count)
    u, sigma, v = low_rank_approx(matrix, k=rank)

    projected_documents = np.dot(matrix.T, u)
    _, document_clustering = cluster(projected_documents, cluster_count)
    document_clusters = [
        [index_to_document[i]["filename"] for (i, x) in enumerate(document_clustering) if x == j]
        for j in range(len(set(document_clustering)))
    ]
    copy_clusters(document_clusters, stories_dir, output_dir)
    return document_clusters


def env_path(var_name: str, default: str) -> Path:
    return Path(os.getenv(var_name, default))


def env_int(var_name: str, default: int) -> int:
    return int(os.getenv(var_name, str(default)))
