#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bz2
import json
from pathlib import Path
import time
from typing import BinaryIO

import faiss
import numpy as np
from tqdm import tqdm

from embedding import OllamaEmbedder
from models import Article, BuildConfig
from utils import configure_logging, read_json, write_json
from chunker import ChunkBuilder
from faiss_helpers import create_index, train_index, clone_trained_empty
from sqlite_store import connect, chunk_rows, INSERT_SQL, finalize_indexes


SCRIPT_DIR = Path(__file__).resolve().parent
RAG_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = RAG_ROOT / "data" / "jsonl" / "ja_wiki.jsonl.bz2"
DEFAULT_INDEX_DIR = RAG_ROOT / "data" / "index"
DEFAULT_BUILD_CONFIG = SCRIPT_DIR / "configs" / "build_config.json"
DEFAULT_INDEX_CONFIG = SCRIPT_DIR / "configs" / "faiss_config.json"
DEFAULT_SAMPLE_VECTORS = SCRIPT_DIR / "sample" / "sample_vectors.f32"
DEFAULT_SAMPLE_MANIFEST = SCRIPT_DIR / "sample" / "sample_vectors.json"


def open_jsonl(path: Path) -> BinaryIO:
    """Open plain or bzip2-compressed JSONL as a binary stream.

    Binary mode keeps tell()/seek() offsets stable so an interrupted
    production build can resume at the last completed shard.
    json.loads() accepts the UTF-8 encoded bytes returned by readline().
    """
    if path.suffix.lower() == ".bz2":
        return bz2.open(path, mode="rb")
    return path.open(mode="rb")


def get_logical_input_size(path: Path) -> int:
    """Return the uncompressed byte size used by tell()/seek().

    A .bz2 file does not store this value in a directly readable header,
    so determining it requires one streaming decompression pass. This is
    only for accurate progress and ETA reporting; the JSONL is not written
    to disk or loaded into memory.
    """
    if path.suffix.lower() != ".bz2":
        return path.stat().st_size

    print(
        "圧縮JSONLの展開後サイズを確認しています"
        "（ファイルは展開・保存しません）..."
    )
    with bz2.open(path, mode="rb") as file:
        file.seek(0, 2)
        return file.tell()


def load_training_vectors(
    path: Path,
    manifest_path: Path,
    dimension: int,
) -> np.ndarray:
    manifest = read_json(manifest_path)
    count = int(manifest["count"])
    vectors = np.memmap(
        path,
        mode="r",
        dtype=np.float32,
        shape=(count, dimension),
    )
    return np.ascontiguousarray(
        vectors,
        dtype=np.float32,
    )


def clean_orphans(
    index_dir: Path,
    completed: set[int],
) -> None:
    for path in index_dir.glob("shard_*"):
        try:
            shard_no = int(
                path.stem.split("_")[1]
            )
        except (IndexError, ValueError):
            continue

        if shard_no not in completed:
            path.unlink(missing_ok=True)


def safe_div(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def format_seconds(
    seconds: float | None,
) -> str | None:
    if seconds is None:
        return None

    seconds = max(0, int(seconds))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def update_progress(
    progress_path: Path,
    *,
    status: str,
    input_path: Path,
    input_size_bytes: int,
    input_offset_bytes: int,
    current_shard: int,
    completed_shards: int,
    completed_articles: int,
    completed_chunks: int,
    session_started: float,
    build_started_epoch: float,
    current_shard_articles: int = 0,
    current_shard_chunks: int = 0,
    message: str = "",
) -> None:
    now_epoch = time.time()
    session_elapsed = time.perf_counter() - session_started
    total_elapsed = now_epoch - build_started_epoch

    progress_ratio = safe_div(
        input_offset_bytes,
        input_size_bytes,
    )

    article_rate = safe_div(
        completed_articles,
        total_elapsed,
    )
    chunk_rate = safe_div(
        completed_chunks,
        total_elapsed,
    )
    byte_rate = safe_div(
        input_offset_bytes,
        total_elapsed,
    )

    remaining_bytes = max(
        0,
        input_size_bytes - input_offset_bytes,
    )

    eta_seconds = (
        safe_div(remaining_bytes, byte_rate)
        if byte_rate > 0
        else None
    )

    payload = {
        "status": status,
        "message": message,
        "input_file": str(input_path.resolve()),
        "input_size_bytes": input_size_bytes,
        "input_offset_bytes": input_offset_bytes,
        "progress_ratio": progress_ratio,
        "progress_percent": progress_ratio * 100,
        "current_shard": current_shard,
        "completed_shards": completed_shards,
        "completed_articles": completed_articles,
        "completed_chunks": completed_chunks,
        "current_shard_articles": current_shard_articles,
        "current_shard_chunks": current_shard_chunks,
        "article_rate_per_second": article_rate,
        "chunk_rate_per_second": chunk_rate,
        "input_rate_mib_per_second": byte_rate / (1024 * 1024),
        "session_elapsed_seconds": session_elapsed,
        "total_elapsed_seconds": total_elapsed,
        "total_elapsed_human": format_seconds(total_elapsed),
        "estimated_remaining_seconds": eta_seconds,
        "estimated_remaining_human": format_seconds(eta_seconds),
        "updated_at_epoch": now_epoch,
    }

    write_json(
        progress_path,
        payload,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input JSONL or JSONL.bz2 (default: ../data/jsonl/ja_wiki.jsonl.bz2)",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Output index directory (default: ../data/index)",
    )
    parser.add_argument(
        "--build-config",
        type=Path,
        default=DEFAULT_BUILD_CONFIG,
    )
    parser.add_argument(
        "--index-config",
        type=Path,
        default=DEFAULT_INDEX_CONFIG,
    )
    parser.add_argument(
        "--sample-vectors",
        type=Path,
        default=DEFAULT_SAMPLE_VECTORS,
    )
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=DEFAULT_SAMPLE_MANIFEST,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
    )
    args = parser.parse_args()

    args.input = args.input.resolve()
    args.index_dir = args.index_dir.resolve()
    args.build_config = args.build_config.resolve()
    args.index_config = args.index_config.resolve()
    args.sample_vectors = args.sample_vectors.resolve()
    args.sample_manifest = args.sample_manifest.resolve()

    if not args.finalize_only and not args.input.is_file():
        parser.error(f"Input JSONL not found: {args.input}")
    if not args.build_config.is_file():
        parser.error(f"Build config not found: {args.build_config}")
    if not args.index_config.is_file():
        parser.error(f"Index config not found: {args.index_config}")
    if not args.finalize_only and not (args.index_dir / "trained_template.faiss").is_file():
        if not args.sample_vectors.is_file():
            parser.error(f"Sample vectors not found: {args.sample_vectors}")
        if not args.sample_manifest.is_file():
            parser.error(f"Sample manifest not found: {args.sample_manifest}")

    args.index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = configure_logging(
        args.index_dir / "build.log"
    )
    progress_path = (
        args.index_dir / "progress.json"
    )

    build_config = BuildConfig(
        **read_json(args.build_config)
    )
    selected = read_json(
        args.index_config
    )

    database_path = (
        args.index_dir / "metadata.sqlite"
    )

    if args.overwrite and database_path.exists():
        for path in args.index_dir.iterdir():
            if path.is_file():
                path.unlink()

    connection = connect(database_path)

    if args.finalize_only:
        finalize_indexes(connection)
        connection.close()

        finalize_input_size = (
            get_logical_input_size(args.input)
            if args.input.exists()
            else 0
        )

        update_progress(
            progress_path,
            status="completed",
            input_path=args.input,
            input_size_bytes=finalize_input_size,
            input_offset_bytes=finalize_input_size,
            current_shard=-1,
            completed_shards=0,
            completed_articles=0,
            completed_chunks=0,
            session_started=time.perf_counter(),
            build_started_epoch=time.time(),
            message="SQLite indexes finalized.",
        )

        print(
            "SQLite indexes finalized."
        )
        return 0

    input_size_bytes = get_logical_input_size(
        args.input
    )
    session_started = time.perf_counter()

    completed_rows = connection.execute(
        "SELECT shard_no, input_end_offset, "
        "article_count, chunk_count "
        "FROM completed_shards "
        "ORDER BY shard_no"
    ).fetchall()

    completed = {
        int(row[0])
        for row in completed_rows
    }

    completed_articles = sum(
        int(row[2])
        for row in completed_rows
    )
    completed_chunks = sum(
        int(row[3])
        for row in completed_rows
    )

    next_shard = (
        0
        if not completed_rows
        else int(completed_rows[-1][0]) + 1
    )
    start_offset = (
        0
        if not completed_rows
        else int(completed_rows[-1][1])
    )

    clean_orphans(
        args.index_dir,
        completed,
    )

    # Preserve the original build start time across resumes.
    previous_progress = None
    if progress_path.exists():
        try:
            previous_progress = read_json(
                progress_path
            )
        except Exception:
            previous_progress = None

    build_started_epoch = (
        float(
            previous_progress.get(
                "build_started_epoch",
                time.time(),
            )
        )
        if previous_progress
        else time.time()
    )

    # Ensure the field exists in future progress writes.
    def write_progress(
        *,
        status: str,
        input_offset_bytes: int,
        current_shard: int,
        completed_shards: int,
        current_shard_articles: int = 0,
        current_shard_chunks: int = 0,
        message: str = "",
    ) -> None:
        update_progress(
            progress_path,
            status=status,
            input_path=args.input,
            input_size_bytes=input_size_bytes,
            input_offset_bytes=input_offset_bytes,
            current_shard=current_shard,
            completed_shards=completed_shards,
            completed_articles=completed_articles,
            completed_chunks=completed_chunks,
            current_shard_articles=current_shard_articles,
            current_shard_chunks=current_shard_chunks,
            session_started=session_started,
            build_started_epoch=build_started_epoch,
            message=message,
        )

        # Append this field without changing update_progress's public signature.
        progress = read_json(progress_path)
        progress["build_started_epoch"] = (
            build_started_epoch
        )
        write_json(
            progress_path,
            progress,
        )

    write_progress(
        status="starting",
        input_offset_bytes=start_offset,
        current_shard=next_shard,
        completed_shards=len(completed),
        message="Preparing production build.",
    )

    template_path = (
        args.index_dir
        / "trained_template.faiss"
    )

    if template_path.exists():
        trained_template = faiss.read_index(
            str(template_path)
        )
    else:
        logger.info(
            "Training production index template"
        )
        write_progress(
            status="training_template",
            input_offset_bytes=start_offset,
            current_shard=next_shard,
            completed_shards=len(completed),
            message=(
                "Training production index "
                "template from sample vectors."
            ),
        )

        training_vectors = (
            load_training_vectors(
                args.sample_vectors,
                args.sample_manifest,
                build_config.embedding_dimension,
            )
        )

        trained_template = create_index(
            build_config.embedding_dimension,
            selected["index_type"],
            int(selected["nlist"]),
            int(
                selected.get(
                    "pq_m",
                    0,
                )
            ),
            int(
                selected.get(
                    "pq_bits",
                    8,
                )
            ),
        )

        train_index(
            trained_template,
            training_vectors,
        )

        faiss.write_index(
            trained_template,
            str(template_path),
        )

        del training_vectors

    builder = ChunkBuilder(build_config)
    embedder = OllamaEmbedder(build_config)

    try:
        with open_jsonl(args.input) as file:
            file.seek(start_offset)

            while True:
                shard_start_offset = (
                    file.tell()
                )
                articles: list[Article] = []

                for _ in range(
                    build_config.shard_articles
                ):
                    line = file.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue

                    obj = json.loads(line)
                    meta = obj.get("meta") or {}
                    line_no = (
                        completed_articles
                        + len(articles)
                        + 1
                    )

                    articles.append(
                        Article(
                            article_id=str(
                                meta.get("id")
                                or (
                                    f"offset:"
                                    f"{shard_start_offset}:"
                                    f"{line_no}"
                                )
                            ),
                            title=str(
                                meta.get("title")
                                or "(無題)"
                            ),
                            url=str(
                                meta.get("url")
                                or ""
                            ),
                            text=str(
                                obj.get("text")
                                or ""
                            ),
                        )
                    )

                if not articles:
                    break

                shard_end_offset = file.tell()

                logger.info(
                    "Building shard %d "
                    "articles=%d offsets=%d..%d",
                    next_shard,
                    len(articles),
                    shard_start_offset,
                    shard_end_offset,
                )

                write_progress(
                    status="chunking",
                    input_offset_bytes=shard_start_offset,
                    current_shard=next_shard,
                    completed_shards=len(completed),
                    current_shard_articles=len(articles),
                    message=(
                        f"Chunking shard "
                        f"{next_shard}."
                    ),
                )

                chunks = []
                for article in tqdm(
                    articles,
                    desc=(
                        f"chunk shard "
                        f"{next_shard}"
                    ),
                    unit="article",
                    dynamic_ncols=True,
                ):
                    chunks.extend(
                        builder.build(article)
                    )

                write_progress(
                    status="embedding",
                    input_offset_bytes=shard_start_offset,
                    current_shard=next_shard,
                    completed_shards=len(completed),
                    current_shard_articles=len(articles),
                    current_shard_chunks=len(chunks),
                    message=(
                        f"Embedding shard "
                        f"{next_shard}."
                    ),
                )

                index = clone_trained_empty(
                    trained_template
                )

                f16_temp = (
                    args.index_dir
                    / (
                        f"shard_"
                        f"{next_shard:06d}"
                        f".f16.tmp"
                    )
                )
                ids_temp = (
                    args.index_dir
                    / (
                        f"shard_"
                        f"{next_shard:06d}"
                        f".ids.tmp"
                    )
                )
                faiss_temp = (
                    args.index_dir
                    / (
                        f"shard_"
                        f"{next_shard:06d}"
                        f".faiss.tmp"
                    )
                )

                f16_file = (
                    open(f16_temp, "wb")
                    if build_config.save_float16
                    else None
                )
                ids_file = (
                    open(ids_temp, "wb")
                    if build_config.save_float16
                    else None
                )

                batch_size = (
                    build_config.embedding_batch_size
                )

                try:
                    for start in tqdm(
                        range(
                            0,
                            len(chunks),
                            batch_size,
                        ),
                        desc=(
                            f"embed shard "
                            f"{next_shard}"
                        ),
                        unit="batch",
                        dynamic_ncols=True,
                    ):
                        batch = chunks[
                            start:start + batch_size
                        ]
                        vectors = (
                            embedder.embed_chunks(
                                batch
                            )
                        )
                        ids = np.asarray(
                            [
                                chunk.chunk_id
                                for chunk in batch
                            ],
                            dtype=np.int64,
                        )

                        index.add_with_ids(
                            vectors,
                            ids,
                        )

                        if (
                            f16_file is not None
                            and ids_file is not None
                        ):
                            vectors.astype(
                                np.float16
                            ).tofile(f16_file)
                            ids.tofile(ids_file)
                finally:
                    if f16_file is not None:
                        f16_file.close()
                    if ids_file is not None:
                        ids_file.close()

                write_progress(
                    status="saving_shard",
                    input_offset_bytes=shard_end_offset,
                    current_shard=next_shard,
                    completed_shards=len(completed),
                    current_shard_articles=len(articles),
                    current_shard_chunks=len(chunks),
                    message=(
                        f"Saving shard "
                        f"{next_shard}."
                    ),
                )

                faiss.write_index(
                    index,
                    str(faiss_temp),
                )

                final_faiss = (
                    args.index_dir
                    / (
                        f"shard_"
                        f"{next_shard:06d}"
                        f".faiss"
                    )
                )
                faiss_temp.replace(
                    final_faiss
                )

                if build_config.save_float16:
                    f16_temp.replace(
                        args.index_dir
                        / (
                            f"shard_"
                            f"{next_shard:06d}"
                            f".f16"
                        )
                    )
                    ids_temp.replace(
                        args.index_dir
                        / (
                            f"shard_"
                            f"{next_shard:06d}"
                            f".ids"
                        )
                    )

                try:
                    connection.execute("BEGIN")
                    connection.executemany(
                        INSERT_SQL,
                        chunk_rows(
                            chunks,
                            next_shard,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO "
                        "completed_shards("
                        "shard_no, "
                        "input_end_offset, "
                        "article_count, "
                        "chunk_count"
                        ") VALUES (?, ?, ?, ?)",
                        (
                            next_shard,
                            shard_end_offset,
                            len(articles),
                            len(chunks),
                        ),
                    )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise

                manifest = {
                    "shard_no": next_shard,
                    "article_count": len(
                        articles
                    ),
                    "chunk_count": len(chunks),
                    "dimension": (
                        build_config
                        .embedding_dimension
                    ),
                    "input_start_offset": (
                        shard_start_offset
                    ),
                    "input_end_offset": (
                        shard_end_offset
                    ),
                    "faiss_file": (
                        final_faiss.name
                    ),
                    "float16_file": (
                        f"shard_"
                        f"{next_shard:06d}"
                        f".f16"
                        if (
                            build_config
                            .save_float16
                        )
                        else None
                    ),
                    "ids_file": (
                        f"shard_"
                        f"{next_shard:06d}"
                        f".ids"
                        if (
                            build_config
                            .save_float16
                        )
                        else None
                    ),
                }

                write_json(
                    args.index_dir
                    / (
                        f"shard_"
                        f"{next_shard:06d}"
                        f".json"
                    ),
                    manifest,
                )

                completed_articles += len(
                    articles
                )
                completed_chunks += len(
                    chunks
                )
                completed.add(next_shard)

                logger.info(
                    "Completed shard %d "
                    "chunks=%d",
                    next_shard,
                    len(chunks),
                )

                write_progress(
                    status="running",
                    input_offset_bytes=shard_end_offset,
                    current_shard=next_shard,
                    completed_shards=len(completed),
                    message=(
                        f"Completed shard "
                        f"{next_shard}."
                    ),
                )

                next_shard += 1

        logger.info(
            "Finalizing SQLite indexes"
        )

        write_progress(
            status="finalizing",
            input_offset_bytes=input_size_bytes,
            current_shard=next_shard,
            completed_shards=len(completed),
            message=(
                "Finalizing SQLite indexes."
            ),
        )

        finalize_indexes(connection)
        connection.close()

        write_json(
            args.index_dir / "config.json",
            {
                "build": (
                    build_config.to_dict()
                ),
                "index": selected,
            },
        )

        write_progress(
            status="completed",
            input_offset_bytes=input_size_bytes,
            current_shard=next_shard,
            completed_shards=len(completed),
            message="Build completed.",
        )

        print("Completed.")
        return 0

    except Exception as exc:
        try:
            write_progress(
                status="failed",
                input_offset_bytes=(
                    start_offset
                ),
                current_shard=next_shard,
                completed_shards=len(completed),
                message=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )
        finally:
            connection.close()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
