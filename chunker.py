from __future__ import annotations
import hashlib, re
from dataclasses import dataclass
from typing import Iterable
from models import Article, BuildConfig, Chunk

_PARAGRAPH_RE = re.compile(r'\n[ \t]*\n+')
_SENTENCE_RE = re.compile(r'(?<=[。！？!?])\s*')
_DISAMBIG = ('曖昧さ回避','以下のものを指す','同名の人物','人名である')

@dataclass(slots=True)
class Unit:
    text: str
    section: str

class ChunkBuilder:
    def __init__(self, config: BuildConfig) -> None:
        self.config = config

    def build(self, article: Article) -> list[Chunk]:
        text = article.text.replace('\r\n','\n').replace('\r','\n').strip()
        if not text:
            return []
        units = list(self._units(text))
        groups = self._pack(units)
        if not groups:
            return []
        ids = [self._stable_id(article.article_id, i) for i in range(len(groups))]
        page_type, quality = self._classify(text)
        out = []
        for i, group in enumerate(groups):
            body = '\n\n'.join(u.text for u in group)
            section = next((u.section for u in reversed(group) if u.section), '')
            out.append(Chunk(
                chunk_id=ids[i], article_id=article.article_id, title=article.title,
                url=article.url, section=section, chunk_no=i, chunk_count=len(groups),
                text=body, prev_chunk_id=ids[i-1] if i else None,
                next_chunk_id=ids[i+1] if i+1 < len(ids) else None,
                page_type=page_type, quality_weight=quality,
            ))
        return out

    def _units(self, text: str) -> Iterable[Unit]:
        section = ''
        for paragraph in _PARAGRAPH_RE.split(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if self._heading(paragraph):
                section = paragraph
                yield Unit(paragraph, section)
                continue
            if len(paragraph) <= self.config.max_chunk_chars:
                yield Unit(paragraph, section)
                continue
            sentences = [s.strip() for s in _SENTENCE_RE.split(paragraph) if s.strip()] or [paragraph]
            for sentence in sentences:
                if len(sentence) <= self.config.max_chunk_chars:
                    yield Unit(sentence, section)
                    continue
                step = max(1, self.config.max_chunk_chars - self.config.overlap_chars)
                start = 0
                while start < len(sentence):
                    fragment = sentence[start:start+self.config.max_chunk_chars].strip()
                    if fragment:
                        yield Unit(fragment, section)
                    if start + self.config.max_chunk_chars >= len(sentence):
                        break
                    start += step

    def _pack(self, units: list[Unit]) -> list[list[Unit]]:
        groups, current = [], []
        for unit in units:
            candidate = current + [unit]
            if current and len('\n\n'.join(x.text for x in candidate)) > self.config.target_chunk_chars:
                groups.append(current)
                current = self._overlap(current)
                candidate = current + [unit]
                current = [unit] if len('\n\n'.join(x.text for x in candidate)) > self.config.max_chunk_chars else candidate
            else:
                current = candidate
        if current:
            groups.append(current)
        if len(groups) >= 2 and len('\n\n'.join(x.text for x in groups[-1])) < self.config.min_chunk_chars:
            merged = groups[-2] + [x for x in groups[-1] if x not in groups[-2]]
            if len('\n\n'.join(x.text for x in merged)) <= self.config.max_chunk_chars:
                groups = groups[:-2] + [merged]
        return groups

    def _overlap(self, group: list[Unit]) -> list[Unit]:
        selected, total = [], 0
        for unit in reversed(group):
            selected.append(unit); total += len(unit.text)
            if total >= self.config.overlap_chars:
                break
        return list(reversed(selected))

    @staticmethod
    def _heading(text: str) -> bool:
        return '\n' not in text and len(text) <= 80 and not text.endswith(('。','！','？','.','!','?'))

    @staticmethod
    def _stable_id(article_id: str, chunk_no: int) -> int:
        digest = hashlib.blake2b(f'{article_id}\0{chunk_no}'.encode('utf-8'), digest_size=8).digest()
        return int.from_bytes(digest, 'big') & 0x7FFF_FFFF_FFFF_FFFF

    @staticmethod
    def _classify(text: str) -> tuple[str, float]:
        return ('disambiguation', 0.82) if any(p in text[:1200] for p in _DISAMBIG) else ('article', 1.0)
