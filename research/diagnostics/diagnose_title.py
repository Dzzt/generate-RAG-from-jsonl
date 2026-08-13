#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import normalize_title


DEFAULT_INDEX_DIR = RAG_ROOT / "data" / "index"


def compact_title(value: str) -> str:
    """Comparison key ignoring width, case, whitespace, punctuation and symbols."""
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(ch for ch in value if ch.isalnum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("title")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    args = parser.parse_args()
    args.index_dir = args.index_dir.resolve()

    database = args.index_dir / "metadata.sqlite"
    if not database.is_file():
        parser.error(f"metadata.sqlite not found: {database}")

    con = sqlite3.connect(database)
    query = args.title.strip()
    normalized = normalize_title(query)
    compact = compact_title(normalized)

    print(f"query={query!r}")
    print(f"normalized={normalized!r}")
    print(f"compact={compact!r}")
    print()

    rows = con.execute(
        "SELECT DISTINCT title, COALESCE(url, ''), normalized_title "
        "FROM chunks WHERE normalized_title=? LIMIT 50",
        (normalized,),
    ).fetchall()

    print("normalized exact:")
    if rows:
        for row in rows:
            print(row)
    else:
        print("(none)")

    print()
    print("FTS candidates:")
    try:
        rows = con.execute(
            "SELECT DISTINCT c.title, COALESCE(c.url, ''), c.normalized_title "
            "FROM title_fts f JOIN chunks c ON c.chunk_id=f.chunk_id "
            "WHERE title_fts MATCH ? LIMIT 100",
            ('"' + normalized.replace('"', '""') + '"',),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        print(f"FTS unavailable: {exc}")
        rows = []

    if rows:
        for title, url, norm in rows:
            print((title, url, norm, compact_title(title)))
    else:
        print("(none)")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
