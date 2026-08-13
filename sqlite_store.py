from __future__ import annotations
import sqlite3
from pathlib import Path
from models import Chunk
from utils import normalize_title

SCHEMA = '''
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
CREATE TABLE IF NOT EXISTS chunks (
 chunk_id INTEGER PRIMARY KEY, article_id TEXT NOT NULL, title TEXT NOT NULL,
 normalized_title TEXT NOT NULL, url TEXT, section TEXT, chunk_no INTEGER NOT NULL,
 chunk_count INTEGER NOT NULL, prev_chunk_id INTEGER, next_chunk_id INTEGER,
 page_type TEXT NOT NULL, quality_weight REAL NOT NULL, vector_shard INTEGER NOT NULL,
 vector_row INTEGER NOT NULL, text TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS completed_shards (
 shard_no INTEGER PRIMARY KEY, input_end_offset INTEGER NOT NULL,
 article_count INTEGER NOT NULL, chunk_count INTEGER NOT NULL,
 completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
'''

def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con

INSERT_SQL = '''INSERT INTO chunks (
chunk_id, article_id, title, normalized_title, url, section, chunk_no, chunk_count,
prev_chunk_id, next_chunk_id, page_type, quality_weight, vector_shard, vector_row, text
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

def chunk_rows(chunks: list[Chunk], shard_no: int) -> list[tuple]:
    return [(
        c.chunk_id,c.article_id,c.title,normalize_title(c.title),c.url,c.section,c.chunk_no,c.chunk_count,
        c.prev_chunk_id,c.next_chunk_id,c.page_type,c.quality_weight,shard_no,row,c.text
    ) for row,c in enumerate(chunks)]

def finalize_indexes(con: sqlite3.Connection) -> None:
    con.execute('CREATE INDEX IF NOT EXISTS idx_chunks_title ON chunks(normalized_title)')
    con.execute('CREATE INDEX IF NOT EXISTS idx_chunks_article ON chunks(article_id, chunk_no)')
    con.execute('CREATE INDEX IF NOT EXISTS idx_chunks_vector ON chunks(vector_shard, vector_row)')
    try:
        con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS title_fts USING fts5(normalized_title, chunk_id UNINDEXED, tokenize='trigram')")
        if con.execute('SELECT count(*) FROM title_fts').fetchone()[0] == 0:
            con.execute('INSERT INTO title_fts(normalized_title, chunk_id) SELECT normalized_title, chunk_id FROM chunks')
    except sqlite3.OperationalError:
        pass
    con.commit()
