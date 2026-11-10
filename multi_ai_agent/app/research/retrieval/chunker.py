"""Split a document into heading-aware chunks so only relevant passages reach the local LLM."""

from __future__ import annotations

from app.research.schemas import Chunk, Document

DEFAULT_CHUNK_CHARS = 900
DEFAULT_OVERLAP = 1  # paragraphs carried over between consecutive chunks


def chunk_document(doc: Document, max_chars: int = DEFAULT_CHUNK_CHARS, overlap: int = DEFAULT_OVERLAP) -> list[Chunk]:
    paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body and not all(line.startswith("#") for line in buf):
            chunks.append(Chunk(doc_url=doc.url, index=len(chunks), text=body, heading=heading))

    for para in paragraphs:
        if para.startswith("#"):
            flush()
            buf = []
            heading = para.lstrip("#").strip()
            buf.append(para)
            continue
        # Hard-split giant paragraphs/tables.
        pieces = [para[i : i + max_chars] for i in range(0, len(para), max_chars)] or [para]
        for piece in pieces:
            if buf and sum(len(b) for b in buf) + len(piece) > max_chars:
                flush()
                carry = [b for b in buf[-overlap:] if not b.startswith("#")] if overlap else []
                buf = ([f"## {heading}"] if heading else []) + carry
            buf.append(piece)
    flush()
    return chunks
