from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(slots=True)
class Article:
    article_id: str
    title: str
    url: str
    text: str
    source: str = "jsonl"


@dataclass(slots=True)
class Chunk:
    chunk_id: int
    article_id: str
    title: str
    url: str
    section: str
    chunk_no: int
    chunk_count: int
    text: str
    prev_chunk_id: int | None = None
    next_chunk_id: int | None = None
    page_type: str = "article"
    quality_weight: float = 1.0

    def document_text(self) -> str:
        section = f"\nセクション: {self.section}" if self.section else ""
        return f"タイトル: {self.title}{section}\n\n{self.text}"


@dataclass(slots=True)
class BuildConfig:
    embedding_model: str = "model_embedding"
    embedding_dimension: int = 768
    query_prefix: str = "検索クエリ: "
    document_prefix: str = "検索文書: "
    target_chunk_chars: int = 1200
    max_chunk_chars: int = 1600
    overlap_chars: int = 200
    min_chunk_chars: int = 80
    embedding_batch_size: int = 16
    shard_articles: int = 50000
    save_float16: bool = True
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
