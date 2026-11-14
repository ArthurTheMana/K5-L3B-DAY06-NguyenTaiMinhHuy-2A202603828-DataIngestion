
import json
from datetime import date
from pathlib import Path
from pypdf import PdfReader

# Resolve paths relative to this script.
starter_dir = Path(__file__).resolve().parents[1]
project_dir = starter_dir.parent
raw_dir = starter_dir / "data" / "raw"
output_file = project_dir / "data" / "corpus_manifest.json"

# Metadata based on the two selected documents.
documents = [
    {
        "source_id": "uow_ai_ml",
        "title": "A Gentle Introduction to Artificial Intelligence and Machine Learning",
        "filename": "ai_ml_fundamentals_uow.pdf",
        "source_url": None,
        "license_or_permission": "TO_VERIFY",
        "version": None,
    },
    {
        "source_id": "colliot_ml_2023",
        "title": "A non-technical introduction to machine learning",
        "filename": "ml_introduction_colliot_2023.pdf",
        "source_url": "https://hal.science/hal-03957125v3",
        "license_or_permission": "CC BY 4.0",
        "version": "hal-03957125v3",
    },
]

manifest = {
    "dataset_name": "AI and Machine Learning Fundamentals",
    "created_at": date.today().isoformat(),
    "documents": [],
}

total_pages = 0
total_readable = 0

for doc in documents:
    pdf_path = raw_dir / doc["filename"]

    if not pdf_path.is_file():
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        char_count = len(text)

        pages.append({
            "page_id": f"{doc['source_id']}_p{page_number:03d}",
            "page_number": page_number,
            "char_count": char_count,
            "readable": char_count >= 100,
            "count_as_valid": False,
        })

    readable = sum(p["readable"] for p in pages)

    record = {
        **doc,
        "source_path_or_url": f"starter/data/raw/{doc['filename']}",
        "accessed_at": None,
        "total_pages": len(pages),
        "readable_pages": readable,
        "valid_content_pages": 0,
        "pages": pages,
    }

    manifest["documents"].append(record)

    total_pages += len(pages)
    total_readable += readable

    print(
        f"{doc['filename']}: "
        f"{len(pages)} pages, {readable} readable"
    )

output_file.parent.mkdir(parents=True, exist_ok=True)

with output_file.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print("-" * 50)
print(f"Total pages: {total_pages}")
print(f"Readable pages: {total_readable}")
print(f"Manifest created: {output_file}")
