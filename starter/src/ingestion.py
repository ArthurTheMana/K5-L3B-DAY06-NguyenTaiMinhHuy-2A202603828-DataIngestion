"""Read and normalize source data. TODO: implement functions."""
import re
import unicodedata
from pathlib import Path

import pymupdf

from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """TODO: extract each physical page; keep its page number and source.
    Detect and record pages with no text. OCR, when needed, is a separate step.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = pymupdf.open(str(path))


    pages: list[DocumentPage] = []

    for page_number, pdf_page in enumerate(reader.pages(), start=1):
        try:
            raw_text = pdf_page.get_text("text") or ""
        except Exception as exc:
            raise RuntimeError(
                f"Failed to extract page {page_number} from {path}"
            ) from exc

        text = normalize_text(raw_text)
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata={
                    "page_id": f"{document_id}_p{page_number:03d}",
                    "char_count": len(text),
                    "no_text": not bool(text),
                    "needs_ocr": not bool(text),
                },
            )
        )

    reader.close()
    return pages


def normalize_text(text: str) -> str:
    """TODO: normalize Unicode and whitespace without discarding useful structure."""
    text = unicodedata.normalize("NFC", text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")

    lines = [
        re.sub(r"[ \t\f\v]+", " ", line).strip()
        for line in text.split("\n")
    ]

    normalized = "\n".join(lines)

    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()

