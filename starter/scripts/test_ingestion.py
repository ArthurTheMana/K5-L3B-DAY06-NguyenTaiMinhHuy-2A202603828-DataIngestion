
import sys
from pathlib import Path

# Make the starter package importable.
starter_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(starter_dir))

from src.ingestion import load_pdf

documents = [
    ("uow_ai_ml", "ai_ml_fundamentals_uow.pdf"),
    ("colliot_ml_2023", "ml_introduction_colliot_2023.pdf"),
]

raw_dir = starter_dir / "data" / "raw"

for document_id, filename in documents:
    path = raw_dir / filename

    pages = load_pdf(
        path=path,
        document_id=document_id,
        source=str(path),
    )

    print(f"\nDocument: {filename}")
    print(f"Total pages: {len(pages)}")
    print(f"First page ID: {pages[0].metadata['page_id']}")
    print(f"First page number: {pages[0].page}")

    empty_pages = sum(
        page.metadata["no_text"]
        for page in pages
    )

    print(f"Pages without text: {empty_pages}")

    assert len(pages) > 0
    assert pages[0].page == 1
    assert pages[0].document_id == document_id

print("\nIngestion test PASSED")
