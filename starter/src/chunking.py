"""Implement fixed-size and structure-aware chunking.
"""

import re
from .models import Chunk, DocumentPage


def _make_chunk(page: DocumentPage, text: str, strategy: str, index: int) -> Chunk:
    """Create a chunk with the original page's source and metadata."""
    metadata = dict(page.metadata)
    metadata.update({
        "page_id": metadata.get("page_id", f"{page.document_id}_p{page.page:03d}"),
        "chunk_index": index,
        "chunk_strategy": strategy,
        "char_count": len(text),
    })
    return Chunk(
        chunk_id=f"{page.document_id}_p{page.page:03d}_{strategy}_{index:03d}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=metadata,
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """TODO (starter): split text into overlapping windows and retain page metadata.
    """
    if size <= 0:
        raise ValueError("size must be greater than 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    text = page.text.strip()
    if not text:
        return []

    step = size - overlap
    chunks: list[Chunk] = []
    for start in range(0, len(text), step):
        part = text[start:start + size]
        if part.strip():
            chunks.append(_make_chunk(page, part, "fixed", len(chunks) + 1))
        if start + size >= len(text):
            break
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """TODO (starter): group paragraphs, split oversized paragraphs and retain metadata.
    """
    if max_size <= 0:
        raise ValueError("max_size must be greater than 0")

    text = page.text.strip()
    if not text:
        return []

    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()
    ]
    chunks: list[Chunk] = []
    current = ""

    def emit() -> None:
        nonlocal current
        if current:
            chunks.append(_make_chunk(page, current, "paragraph", len(chunks) + 1))
            current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_size:
            emit()
            for start in range(0, len(paragraph), max_size):
                part = paragraph[start:start + max_size]
                chunks.append(_make_chunk(page, part, "paragraph", len(chunks) + 1))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_size:
            current = candidate
        else:
            emit()
            current = paragraph

    emit()
    return chunks
