
import json
from pathlib import Path

manifest_path = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "corpus_manifest.json"
)

with manifest_path.open("r", encoding="utf-8") as f:
    manifest = json.load(f)

# Candidate pages based on our initial document review.
candidates = {
    "uow_ai_ml": set(range(5, 51)) - {11},
    "colliot_ml_2023": set(range(2, 25)),
}

for doc in manifest["documents"]:
    source_id = doc["source_id"]
    selected = candidates.get(source_id, set())

    for page in doc["pages"]:
        page["count_as_valid"] = (
            page["page_number"] in selected
            and page["readable"]
        )

    doc["valid_content_pages"] = sum(
        page["count_as_valid"] for page in doc["pages"]
    )

    print(
        f"{source_id}: "
        f"{doc['valid_content_pages']} candidate pages"
    )

with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print("Manifest updated.")
