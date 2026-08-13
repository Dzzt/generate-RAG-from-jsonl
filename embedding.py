from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
import ollama

from models import BuildConfig, Chunk


logger = logging.getLogger("rag")


class OllamaEmbedder:
    def __init__(self, config: BuildConfig) -> None:
        self.config = config

    def embed_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        texts = [
            f"{self.config.document_prefix}{chunk.document_text()}"
            for chunk in chunks
        ]
        labels = [
            (
                f"title={chunk.title!r} "
                f"chunk={chunk.chunk_no + 1}/{chunk.chunk_count} "
                f"chars={len(text)}"
            )
            for chunk, text in zip(chunks, texts)
        ]
        return self.embed_texts(texts, labels)

    def embed_query(self, query: str) -> np.ndarray:
        text = f"{self.config.query_prefix}{query.strip()}"
        return self._adaptive([text], [f"query chars={len(text)}"])[0]

    def embed_texts(
        self,
        texts: Sequence[str],
        labels: Sequence[str] | None = None,
    ) -> np.ndarray:
        labels = labels or [
            f"item={index} chars={len(text)}"
            for index, text in enumerate(texts)
        ]
        parts = []
        for start in range(0, len(texts), self.config.embedding_batch_size):
            parts.append(
                self._adaptive(
                    list(texts[start:start + self.config.embedding_batch_size]),
                    list(labels[start:start + self.config.embedding_batch_size]),
                )
            )
        return np.vstack(parts).astype(np.float32, copy=False)

    def _adaptive(self, texts: list[str], labels: list[str]) -> np.ndarray:
        try:
            return self._request(texts)
        except Exception as exc:
            if not self._context_error(exc):
                raise RuntimeError(
                    f"Embedding failed: {' | '.join(labels)} / {exc}"
                ) from exc

            if len(texts) > 1:
                mid = len(texts) // 2
                logger.warning(
                    "Embedding batch context overflow; split %d -> %d + %d",
                    len(texts),
                    mid,
                    len(texts) - mid,
                )
                return np.vstack(
                    (
                        self._adaptive(texts[:mid], labels[:mid]),
                        self._adaptive(texts[mid:], labels[mid:]),
                    )
                )

            original = texts[0]
            length = int(len(original) * 0.75)
            while length >= 256:
                candidate = original[:length].rstrip()
                try:
                    vector = self._request([candidate])
                    logger.warning(
                        "Shortened overlong embedding input: %s original=%d used=%d",
                        labels[0],
                        len(original),
                        len(candidate),
                    )
                    return vector
                except Exception as inner:
                    if not self._context_error(inner):
                        raise
                length = int(length * 0.75)

            raise RuntimeError(
                f"Could not shorten input: {labels[0]}"
            ) from exc

    def _request(self, texts: list[str]) -> np.ndarray:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = ollama.embed(
                    model=self.config.embedding_model,
                    input=texts,
                )
                matrix = np.asarray(
                    response["embeddings"],
                    dtype=np.float32,
                )
                expected = (
                    len(texts),
                    self.config.embedding_dimension,
                )
                if matrix.shape != expected:
                    raise ValueError(
                        f"Unexpected embedding shape: "
                        f"{matrix.shape}, expected {expected}"
                    )

                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                if np.any(norms == 0):
                    raise ValueError("Zero embedding returned")

                return np.ascontiguousarray(
                    matrix / norms,
                    dtype=np.float32,
                )
            except Exception as exc:
                last_error = exc
                if self._context_error(exc):
                    raise
                time.sleep(attempt + 1)

        assert last_error is not None
        raise last_error

    @staticmethod
    def _context_error(exc: Exception) -> bool:
        message = str(exc).casefold()
        return (
            "context length" in message
            or "input length exceeds" in message
            or "too long" in message
        )
