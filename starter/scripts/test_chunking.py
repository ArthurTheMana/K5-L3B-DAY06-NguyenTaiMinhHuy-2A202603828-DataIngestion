"""Smoke-test the chunking functions on both PDFs."""
import sys
from pathlib import Path

starter = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(starter))

from src.ingestion import load_pdf
from src.chunking import fixed_size, by_paragraph

raw_dir = starter / 'data' / 'raw'
documents = [
    ('uow_ai_ml', 'ai_ml_fundamentals_uow.pdf'),
    ('colliot_ml_2023', 'ml_introduction_colliot_2023.pdf'),
]

for doc_id, filename in documents:
    pdf = raw_dir / filename
    pages = load_pdf(pdf, document_id=doc_id, source=str(pdf))

    fixed = [chunk for page in pages for chunk in fixed_size(page, 500, 50)]
    paragraph = [chunk for page in pages for chunk in by_paragraph(page, 500)]

    assert all(0 < len(chunk.text) <= 500 for chunk in fixed + paragraph)
    assert all(chunk.document_id == doc_id for chunk in fixed + paragraph)
    assert len({chunk.chunk_id for chunk in fixed}) == len(fixed)
    assert len({chunk.chunk_id for chunk in paragraph}) == len(paragraph)

    print(f'{filename}: {len(pages)} pages')
    print(f'  Strategy A (fixed): {len(fixed)} chunks')
    print(f'  Strategy B (paragraph): {len(paragraph)} chunks')

print('\nChunking test PASSED')
