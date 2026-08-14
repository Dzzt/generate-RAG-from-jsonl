#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import time
import zlib
from pathlib import Path


CHUNK_COLUMNS = (
    "chunk_id",
    "article_id",
    "title",
    "normalized_title",
    "url",
    "section",
    "chunk_no",
    "chunk_count",
    "prev_chunk_id",
    "next_chunk_id",
    "page_type",
    "quality_weight",
    "vector_shard",
    "vector_row",
    "text",
)

CREATE_SCHEMA = """
CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY,
    article_id TEXT NOT NULL,
    title TEXT NOT NULL,
    normalized_title TEXT NOT NULL,
    url TEXT,
    section TEXT,
    chunk_no INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    prev_chunk_id INTEGER,
    next_chunk_id INTEGER,
    page_type TEXT NOT NULL,
    quality_weight REAL NOT NULL,
    vector_shard INTEGER NOT NULL,
    vector_row INTEGER NOT NULL,
    text BLOB NOT NULL
);

CREATE TABLE completed_shards (
    shard_no INTEGER PRIMARY KEY,
    input_end_offset INTEGER NOT NULL,
    article_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

INSERT_CHUNK = (
    "INSERT INTO chunks ("
    + ", ".join(CHUNK_COLUMNS)
    + ") VALUES ("
    + ", ".join("?" for _ in CHUNK_COLUMNS)
    + ")"
)


def human_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(value)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def human_time(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def validate_source(con: sqlite3.Connection) -> None:
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'"
    ).fetchone()
    if row is None:
        raise RuntimeError("source database has no 'chunks' table")

    columns = {row[1] for row in con.execute("PRAGMA table_info(chunks)")}
    missing = set(CHUNK_COLUMNS) - columns
    if missing:
        raise RuntimeError(
            "source chunks table is missing columns: "
            + ", ".join(sorted(missing))
        )


def finalize_indexes(con: sqlite3.Connection) -> None:
    print()
    print("Creating SQLite indexes...")
    con.execute(
        "CREATE INDEX idx_chunks_title "
        "ON chunks(normalized_title)"
    )
    con.execute(
        "CREATE INDEX idx_chunks_article "
        "ON chunks(article_id, chunk_no)"
    )
    con.execute(
        "CREATE INDEX idx_chunks_vector "
        "ON chunks(vector_shard, vector_row)"
    )
    con.commit()

    print("Creating title_fts...")
    try:
        con.execute(
            "CREATE VIRTUAL TABLE title_fts USING fts5("
            "normalized_title, chunk_id UNINDEXED, tokenize='trigram'"
            ")"
        )
        con.execute(
            "INSERT INTO title_fts(normalized_title, chunk_id) "
            "SELECT normalized_title, chunk_id FROM chunks"
        )
        con.commit()
    except sqlite3.OperationalError as exc:
        print(f"WARNING: title_fts could not be created: {exc}")

    try:
        con.execute("PRAGMA optimize")
    except sqlite3.OperationalError:
        pass
    con.commit()


def copy_completed_shards(
    source: sqlite3.Connection,
    dest: sqlite3.Connection,
) -> None:
    exists = source.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name='completed_shards'"
    ).fetchone()

    if not exists:
        return

    rows = source.execute(
        "SELECT shard_no, input_end_offset, article_count, "
        "chunk_count, completed_at "
        "FROM completed_shards ORDER BY shard_no"
    ).fetchall()

    if rows:
        dest.executemany(
            "INSERT INTO completed_shards("
            "shard_no, input_end_offset, article_count, "
            "chunk_count, completed_at"
            ") VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        dest.commit()


def convert(
    source_path: Path,
    dest_path: Path,
    *,
    level: int,
    batch_size: int,
) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(f"source database not found: {source_path}")

    if source_path.resolve() == dest_path.resolve():
        raise ValueError("source and destination must be different files")

    if dest_path.exists():
        raise FileExistsError(
            f"destination already exists: {dest_path}\n"
            "Delete it or choose another --output path."
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    source_size = source_path.stat().st_size
    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    dest = sqlite3.connect(dest_path)

    try:
        validate_source(source)

        print(f"Source : {source_path}")
        print(f"Output : {dest_path}")
        print(f"Source size : {human_bytes(source_size)}")
        print(f"zlib level  : {level}")
        print(f"batch size  : {batch_size:,}")
        print()

        print("Counting chunks...")
        total = int(source.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        print(f"Total chunks: {total:,}")

        dest.execute("PRAGMA journal_mode=DELETE")
        dest.execute("PRAGMA synchronous=NORMAL")
        dest.execute("PRAGMA temp_store=MEMORY")
        dest.executescript(CREATE_SCHEMA)
        dest.commit()

        select_sql = (
            "SELECT " + ", ".join(CHUNK_COLUMNS)
            + " FROM chunks ORDER BY chunk_id"
        )
        cursor = source.execute(select_sql)

        processed = 0
        original_text_bytes = 0
        compressed_text_bytes = 0
        started = time.perf_counter()
        last_report = started

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            converted = []
            for row in rows:
                values = list(row)
                text = values[-1]

                if isinstance(text, str):
                    raw = text.encode("utf-8")
                elif isinstance(text, (bytes, bytearray, memoryview)):
                    raw = bytes(text)
                    try:
                        raw = zlib.decompress(raw)
                    except zlib.error:
                        pass
                else:
                    raw = str(text).encode("utf-8")

                blob = zlib.compress(raw, level)
                values[-1] = sqlite3.Binary(blob)

                original_text_bytes += len(raw)
                compressed_text_bytes += len(blob)
                converted.append(tuple(values))

            dest.executemany(INSERT_CHUNK, converted)
            dest.commit()

            processed += len(rows)
            now = time.perf_counter()

            if now - last_report >= 1.0 or processed == total:
                elapsed = now - started
                rate = processed / elapsed if elapsed > 0 else 0.0
                remaining = (
                    (total - processed) / rate
                    if rate > 0
                    else None
                )
                ratio = (
                    compressed_text_bytes / original_text_bytes
                    if original_text_bytes
                    else 0.0
                )
                print(
                    f"\r{processed:,}/{total:,} "
                    f"({processed / total * 100:6.2f}%)  "
                    f"{rate:,.0f} chunks/s  "
                    f"ETA {human_time(remaining)}  "
                    f"text {ratio * 100:5.1f}%   ",
                    end="",
                    flush=True,
                )
                last_report = now

        print()
        print()
        print("Copying completed_shards...")
        copy_completed_shards(source, dest)

        finalize_indexes(dest)

        print()
        print("Validating...")
        source_count = int(source.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        dest_count = int(dest.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

        if source_count != dest_count:
            raise RuntimeError(
                f"row-count mismatch: source={source_count}, dest={dest_count}"
            )

        sample = dest.execute(
            "SELECT chunk_id, text FROM chunks ORDER BY chunk_id LIMIT 1"
        ).fetchone()

        if sample is not None:
            value = sample[1]
            if not isinstance(value, (bytes, bytearray, memoryview)):
                raise RuntimeError(
                    "validation failed: destination text is not stored as BLOB"
                )
            zlib.decompress(bytes(value)).decode("utf-8")

        dest.close()
        dest = None

        output_size = dest_path.stat().st_size
        elapsed = time.perf_counter() - started

        print("Completed.")
        print(f"Elapsed      : {human_time(elapsed)}")
        print(f"Source DB    : {human_bytes(source_size)}")
        print(f"Output DB    : {human_bytes(output_size)}")
        print(f"DB ratio     : {output_size / source_size * 100:.1f}%")
        print(
            f"Text ratio   : "
            f"{compressed_text_bytes / original_text_bytes * 100:.1f}%"
        )
        print(f"Space saved  : {human_bytes(source_size - output_size)}")

    finally:
        source.close()
        if dest is not None:
            dest.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a copy of metadata.sqlite whose chunks.text values "
            "are zlib-compressed BLOBs."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(r"..\data\index\metadata.sqlite"),
        help="source metadata.sqlite",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"..\data\index\metadata_compressed.sqlite"),
        help="destination SQLite file",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=6,
        choices=range(0, 10),
        metavar="0..9",
        help="zlib compression level (default: 6)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="rows copied per transaction (default: 10000)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")

    convert(
        args.input.resolve(),
        args.output.resolve(),
        level=args.level,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
