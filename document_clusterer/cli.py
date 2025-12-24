from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from . import clean_directory, cluster_documents, save_documents
from .cleaning import CleaningOptions, env_path as cleaning_env_path
from .model import env_path as model_env_path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Document clusterer utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean_parser = subparsers.add_parser("clean", help="Clean raw stories and write JSON output")
    clean_parser.add_argument(
        "--stories-dir",
        type=Path,
        default=cleaning_env_path("STORIES_DIR", "data/cnn-stories"),
        help="Directory containing story text files (default: %(default)s or STORIES_DIR)",
    )
    clean_parser.add_argument(
        "--word-list",
        type=Path,
        default=cleaning_env_path("WORD_LIST_PATH", "data/one-grams.txt"),
        help="Path to newline-delimited word list (default: %(default)s or WORD_LIST_PATH)",
    )
    clean_parser.add_argument(
        "--stop-words",
        type=Path,
        default=os.getenv("STOP_WORDS_PATH"),
        help="Optional stop-word file to merge with defaults (or override if --no-default-stopwords)",
    )
    clean_parser.add_argument(
        "--extra-stopword",
        action="append",
        default=[],
        help="Additional stop words (can be repeated)",
    )
    clean_parser.add_argument(
        "--pipeline",
        choices=["nltk", "spacy"],
        default=os.getenv("CLEANING_PIPELINE", "nltk"),
        help="Tokenization pipeline to use (default: %(default)s or CLEANING_PIPELINE)",
    )
    clean_parser.add_argument(
        "--spacy-model",
        default=os.getenv("SPACY_MODEL", "en_core_web_sm"),
        help="spaCy model to load when using the spaCy pipeline (default: %(default)s or SPACY_MODEL)",
    )
    clean_parser.add_argument(
        "--min-token-length",
        type=int,
        default=int(os.getenv("MIN_TOKEN_LENGTH", "3")),
        help="Minimum token length to keep (default: %(default)s or MIN_TOKEN_LENGTH)",
    )
    clean_parser.add_argument(
        "--no-lowercase",
        action="store_false",
        dest="lowercase",
        help="Disable lowercasing during cleaning",
    )
    clean_parser.add_argument(
        "--keep-urls",
        action="store_false",
        dest="strip_urls",
        help="Keep URLs instead of removing them",
    )
    clean_parser.add_argument(
        "--keep-numbers",
        action="store_false",
        dest="strip_numbers",
        help="Keep numbers instead of removing them",
    )
    clean_parser.add_argument(
        "--no-default-stopwords",
        action="store_false",
        dest="include_default_stopwords",
        help="Do not use NLTK's default English stopword list",
    )
    clean_parser.add_argument(
        "--output",
        type=Path,
        default=cleaning_env_path("OUTPUT_JSON", "all_stories.json"),
        help="Destination for cleaned JSON output (default: %(default)s or OUTPUT_JSON)",
    )

    cluster_parser = subparsers.add_parser("cluster", help="Cluster cleaned stories")
    cluster_parser.add_argument(
        "--input-file",
        type=Path,
        default=model_env_path("INPUT_JSON", "all_stories.json"),
        help="Path to cleaned JSON input (default: %(default)s or INPUT_JSON)",
    )
    cluster_parser.add_argument(
        "--stories-dir",
        type=Path,
        default=model_env_path("STORIES_DIR", "data/cnn-stories"),
        help="Directory containing original story text files (default: %(default)s or STORIES_DIR)",
    )
    cluster_parser.add_argument(
        "--output-dir",
        type=Path,
        default=model_env_path("CLUSTER_OUTPUT_DIR", "clusteredDocuments"),
        help="Directory to write clustered documents (default: %(default)s or CLUSTER_OUTPUT_DIR)",
    )
    cluster_parser.add_argument(
        "--clusters",
        type=int,
        default=int(os.getenv("CLUSTER_COUNT", "10")),
        help="Number of clusters to generate when using KMeans (default: %(default)s or CLUSTER_COUNT)",
    )
    cluster_parser.add_argument(
        "--cluster-method",
        choices=["kmeans", "hdbscan"],
        default=os.getenv("CLUSTER_METHOD", "kmeans"),
        help="Clustering algorithm to use (default: %(default)s or CLUSTER_METHOD)",
    )
    cluster_parser.add_argument(
        "--model-name",
        default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        help="SentenceTransformers model to encode documents (default: %(default)s or EMBEDDING_MODEL)",
    )
    cluster_parser.add_argument(
        "--kmeans-random-state",
        type=int,
        default=int(os.getenv("KMEANS_RANDOM_STATE", "42")),
        help="Random seed for KMeans (default: %(default)s or KMEANS_RANDOM_STATE)",
    )
    cluster_parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "5")),
        help="Minimum cluster size for HDBSCAN (default: %(default)s or HDBSCAN_MIN_CLUSTER_SIZE)",
    )
    cluster_parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=int(os.getenv("HDBSCAN_MIN_SAMPLES")) if os.getenv("HDBSCAN_MIN_SAMPLES") else None,
        help="Minimum samples for HDBSCAN (default: %(default)s or HDBSCAN_MIN_SAMPLES)",
    )
    cluster_parser.add_argument(
        "--reduction",
        choices=["umap", "pca", "none"],
        default=os.getenv("REDUCTION_METHOD", "umap"),
        help="Dimensionality reduction for visualization (default: %(default)s or REDUCTION_METHOD)",
    )
    cluster_parser.add_argument(
        "--reduction-dim",
        type=int,
        default=int(os.getenv("REDUCTION_DIM", "2")),
        help="Output dimensions for visualization (default: %(default)s or REDUCTION_DIM)",
    )
    cluster_parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=int(os.getenv("UMAP_NEIGHBORS", "15")),
        help="Number of neighbors for UMAP (default: %(default)s or UMAP_NEIGHBORS)",
    )
    cluster_parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=float(os.getenv("UMAP_MIN_DIST", "0.1")),
        help="Minimum distance for UMAP (default: %(default)s or UMAP_MIN_DIST)",
    )
    cluster_parser.add_argument(
        "--summary-terms",
        type=int,
        default=int(os.getenv("SUMMARY_TERMS", "10")),
        help="Top terms per cluster to include in summaries (default: %(default)s or SUMMARY_TERMS)",
    )
    cluster_parser.add_argument(
        "--assignments-basename",
        default=os.getenv("ASSIGNMENTS_BASENAME", "cluster_assignments"),
        help="Base filename for assignment JSON/CSV outputs (default: %(default)s or ASSIGNMENTS_BASENAME)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "clean":
        options = CleaningOptions(
            lowercase=args.lowercase,
            strip_urls=args.strip_urls,
            strip_numbers=args.strip_numbers,
            pipeline=args.pipeline,
            spacy_model=args.spacy_model,
            min_token_length=args.min_token_length,
            include_default_stopwords=args.include_default_stopwords,
        )

        documents = clean_directory(
            args.stories_dir,
            args.word_list,
            stop_words_path=args.stop_words,
            extra_stopwords=args.extra_stopword,
            options=options,
        )
        save_documents(documents, args.output)
    elif args.command == "cluster":
        reduction_method = args.reduction
        if reduction_method == "none":
            reduction_method = None

        hdbscan_min_samples = (
            None if args.hdbscan_min_samples is None else int(args.hdbscan_min_samples)
        )
        cluster_documents(
            data_path=args.input_file,
            stories_dir=args.stories_dir,
            output_dir=args.output_dir,
            model_name=args.model_name,
            cluster_method=args.cluster_method,
            cluster_count=args.clusters if args.cluster_method == "kmeans" else None,
            kmeans_random_state=args.kmeans_random_state,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
            hdbscan_min_samples=hdbscan_min_samples,
            reduction_method=reduction_method,
            reduction_dim=args.reduction_dim,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            summary_top_n=args.summary_terms,
            assignments_basename=args.assignments_basename,
        )
    else:
        parser.error("No command provided")


def clean_cli() -> None:
    argv = sys.argv[1:]
    main(["clean", *argv])


def cluster_cli() -> None:
    argv = sys.argv[1:]
    main(["cluster", *argv])


if __name__ == "__main__":
    main()
