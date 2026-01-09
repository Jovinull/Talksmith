from __future__ import annotations

import re
from pathlib import Path

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WS_RE = re.compile(r"\s+")


def split_sentences(text: str) -> list[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = _SENT_SPLIT_RE.split(text)

    sentences: list[str] = []
    for p in parts:
        s = _WS_RE.sub(" ", p).strip()
        if len(s) >= 3:
            sentences.append(s)
    return sentences


def make_chunks(sentences: list[str], chunk_size: int = 5, overlap: int = 2) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size precisa ser >= 1")

    if overlap < 0:
        overlap = 0

    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    step = max(1, chunk_size - overlap)

    chunks: list[str] = []
    i = 0
    n = len(sentences)

    while i < n:
        window = sentences[i : i + chunk_size]
        if not window:
            break

        chunk = " ".join(window).strip()
        if len(chunk) >= 30:
            chunks.append(chunk)

        if i + chunk_size >= n:
            break

        i += step

    return chunks


def load_chunks_from_file(path: str | Path, chunk_size: int = 5, overlap: int = 2) -> list[str]:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    sentences = split_sentences(raw)
    return make_chunks(sentences, chunk_size=chunk_size, overlap=overlap)
