
from pathlib import Path
from pypdf import PdfReader

# Folder containing our PDF documents
folder = Path("data/raw")

total_pages = 0
readable_pages = 0

for pdf in sorted(folder.glob("*.pdf")):
    reader = PdfReader(str(pdf))
    pages = len(reader.pages)

    valid = 0

    for page in reader.pages:
        text = page.extract_text() or ""

        if len(text.strip()) >= 100:
            valid += 1

    total_pages += pages
    readable_pages += valid

    print(f"{pdf.name}: {pages} pages, {valid} readable")

print("-" * 50)
print(f"Total pages: {total_pages}")
print(f"Readable pages: {readable_pages}")

