from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import clean_directory, cluster_documents, save_documents
from .cleaning import env_path as cleaning_env_path
from .model import env_int as model_env_int, env_path as model_env_path

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
        default=model_env_int("CLUSTER_COUNT", 10),
        help="Number of clusters to generate (default: %(default)s or CLUSTER_COUNT)",
    )
    cluster_parser.add_argument(
        "--rank",
        type=int,
        default=model_env_int("SVD_RANK", 10),
        help="Rank to use for SVD approximation (default: %(default)s or SVD_RANK)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "clean":
        documents = clean_directory(args.stories_dir, args.word_list)
        save_documents(documents, args.output)
    elif args.command == "cluster":
        cluster_documents(
            data_path=args.input_file,
            stories_dir=args.stories_dir,
            output_dir=args.output_dir,
            cluster_count=args.clusters,
            rank=args.rank,
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
