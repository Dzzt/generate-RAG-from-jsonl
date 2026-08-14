import sqlite3
import time
import zlib
from pathlib import Path

db_path = Path(r"..\data\index\metadata.sqlite").resolve()

SAMPLE_COUNT = 100_000
ZLIB_LEVELS = [1, 6, 9]

con = sqlite3.connect(db_path)

print(f"DB: {db_path}")
print(f"sample chunks: {SAMPLE_COUNT:,}")
print()

print("Loading sample texts...")

start = time.perf_counter()

rows = con.execute("""
    SELECT text
    FROM chunks
    LIMIT ?
""", (SAMPLE_COUNT,)).fetchall()

load_sec = time.perf_counter() - start

texts = [
    (row[0] or "").encode("utf-8")
    for row in rows
]

original_bytes = sum(len(x) for x in texts)

print(f"loaded       : {len(texts):,} chunks")
print(f"original size: {original_bytes / 1024**2:.2f} MiB")
print(f"load time    : {load_sec:.3f} sec")
print()

for level in ZLIB_LEVELS:
    print(f"=== zlib level {level} ===")

    start = time.perf_counter()

    compressed = [
        zlib.compress(text, level)
        for text in texts
    ]

    compress_sec = time.perf_counter() - start

    compressed_bytes = sum(len(x) for x in compressed)

    start = time.perf_counter()

    restored = [
        zlib.decompress(blob)
        for blob in compressed
    ]

    decompress_sec = time.perf_counter() - start

    # 念のため元に戻ることを確認
    assert restored == texts

    ratio = compressed_bytes / original_bytes

    print(
        f"compressed size : "
        f"{compressed_bytes / 1024**2:.2f} MiB"
    )

    print(
        f"ratio           : "
        f"{ratio * 100:.1f}%"
    )

    print(
        f"space saved     : "
        f"{(1 - ratio) * 100:.1f}%"
    )

    print(
        f"compress time   : "
        f"{compress_sec:.3f} sec"
    )

    print(
        f"decompress time : "
        f"{decompress_sec:.3f} sec"
    )

    print()

con.close()