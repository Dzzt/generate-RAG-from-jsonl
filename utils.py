from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path


def configure_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rag")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return __import__("json").load(file)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temp_path.replace(path)


_SPACE_RE = re.compile(r"\s+")


def normalize_title(value: str) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        value,
    ).casefold().strip()
    return _SPACE_RE.sub(" ", normalized)
