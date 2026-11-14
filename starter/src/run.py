"""Suggested pipeline entry point; TODO: implement the orchestration."""
"""Load PDF documents, build and store text embeddings, and evaluate retrieval strategies."""
import argparse
import csv
import json
import re
import os
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .store import PgVectorStore

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DOC_TYPES = {"uow_ai_ml": "lecture_slides", "colliot_ml_2023": "book_chapter"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clean(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).casefold()).strip()


def load_inputs():
    manifest = read_json(ROOT / "data/corpus_manifest.json")
    questions = read_json(ROOT / "data/benchmark/benchmark_20_questions.json")
    docs = manifest["documents"]
    sources = {doc["source_id"] for doc in docs}
    if len(sources) != len(docs) or sources - DOC_TYPES.keys():
        raise ValueError("Duplicate or unrecognized document IDs in manifest")
    required = {"id", "split", "question", "expected_answer", "gold_source_id", "gold_page", "evidence_quote"}
    counts = Counter(q.get("split") for q in questions)
    if (len({q.get("id") for q in questions}) != len(questions)
            or counts["dev"] < 15 or counts["test"] < 5
            or sum(counts.values()) != counts["dev"] + counts["test"]):
        raise ValueError("Benchmark requires unique IDs, 15 dev and 5 test questions")
    for q in questions:
        if (required - q.keys() or q["gold_source_id"] not in sources
                or int(q["gold_page"]) < 1
                or not all(str(q[k]).strip() for k in ("question", "expected_answer", "evidence_quote"))
                or (q.get("filter") is not None and not isinstance(q["filter"], dict))):
            raise ValueError(f"Invalid benchmark question: {q.get('id')}")
    filtered = [q for q in questions if q.get("filter")]
    if len(filtered) < 3 or not any(q["split"] == "test" for q in filtered):
        raise ValueError("Benchmark requires 3 filtered questions, including one test question")

    pages, corpus, confirmed = [], {}, 0
    for doc in docs:
        source_id = doc["source_id"]
        pdf = ROOT / "data/raw" / doc["filename"]
        extracted = load_pdf(pdf, document_id=source_id,
                             source=doc.get("source_url") or str(pdf.relative_to(ROOT)))
        if len(extracted) != doc["total_pages"]:
            raise ValueError(f"Page count differs from manifest: {source_id}")
        entries = {entry["page_number"]: entry for entry in doc["pages"]}
        valid = 0
        for page in extracted:
            entry = entries.get(page.page)
            if entry is None or (entry.get("count_as_valid") and not page.text.strip()):
                raise ValueError(f"Invalid page in manifest: {source_id} p{page.page}")
            valid += bool(entry.get("count_as_valid"))
            page.metadata.update({"source_id": source_id, "document_type": DOC_TYPES[source_id],
                                  "verified_content_page": bool(entry.get("count_as_valid"))})
            if page.text.strip():
                pages.append(page)
        confirmed += valid
        corpus[source_id] = {"pdf_pages": len(extracted), "confirmed_content_pages": valid,
                             "license_or_permission": doc.get("license_or_permission")}
    if confirmed < 30:
        raise ValueError(f"Only {confirmed} confirmed content pages; at least 30 required")
    if any(doc.get("license_or_permission") in (None, "", "TO_VERIFY") for doc in docs):
        print("WARNING: Verify document permission before sharing PDFs")

    lookup = {(p.document_id, p.page): p.text for p in pages}
    quote_checks = {}
    for q in questions:
        location = (q["gold_source_id"], int(q["gold_page"]))
        if location not in lookup:
            raise ValueError(f"Missing gold source page for question {q['id']}")
        quote_checks[q["id"]] = clean(q["evidence_quote"]) in clean(lookup[location])
    return pages, questions, corpus, quote_checks


def write_csv(name, header, rows):
    with (RESULTS / name).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_reviews(pages, chunks, traces):
    # Same five confirmed pages from each document for all strategies.
    locations = [location for doc_id in sorted(DOC_TYPES)
                 for location in [(p.document_id, p.page) for p in pages
                                  if p.document_id == doc_id and p.metadata["verified_content_page"]][:5]]
    by_location = {(s, c.document_id, c.page): c
                   for s, group in chunks.items() for c in reversed(group)}
    write_csv("manual_chunk_coherence.csv",
              ["strategy", "document_id", "page", "chunk_id", "chunk_text", "coherence_0_1_2", "reason"],
              ([s, d, p, c.chunk_id if c else "", c.text if c else "", "", ""]
               for s in ("A", "B", "C") for d, p in locations
               for c in [by_location.get((s, d, p))]))
    write_csv("manual_test_grounding.csv",
              ["strategy", "question_id", "question", "retrieved_top3_ids", "grounding_0_1_2",
               "source_page_metadata_correct_y_n", "reason"],
              ([t["strategy"], t["id"], t["question"],
                " | ".join(h["chunk_id"] for h in t["hits"]), "", "", ""]
               for t in traces if t["split"] == "test"))


def main() -> None:
    """TODO:

    1. Load the corpus and benchmark; validate at least 30 content pages.
    2. Build strategies A (fixed), B (paragraph) and C (B + metadata filter).
    3. Embed with the same model and store vectors/metadata per strategy.
    4. Run the same dev/test questions and save full top-3 traces.
    5. Report retrieval metrics and prepare samples for manual assessment.

    Follow the five dimensions in ../../BENCHMARK.md.
    """
    parser = argparse.ArgumentParser(description="Build embeddings and evaluate retrieval strategies")
    parser.add_argument("--check", action="store_true", help="Check corpus and benchmark without database writes")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    pages, questions, corpus, quotes = load_inputs()
    mismatches = [qid for qid, found in quotes.items() if not found]
    print(f"Gold quotes on source pages: {len(quotes) - len(mismatches)}/{len(quotes)}")
    if mismatches:
        print("WARNING: Review unmatched gold quotes:", ", ".join(mismatches))
    if args.check:
        print("Corpus and benchmark checks complete")
        return

    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is missing from .env")
    started = time.perf_counter()
    chunks = {
        "A": [c for p in pages for c in fixed_size(p, 500, 50)],
        "B": [c for p in pages for c in by_paragraph(p, 500)],
    }
    if not all(chunks.values()):
        raise ValueError("Chunking produced no chunks")
    chunks["C"] = chunks["B"]  # C reuses B vectors; filtering happens during retrieval.

    embedder = MyEmbedder(os.getenv("EMBEDDING_MODEL") or MODEL)
    store = PgVectorStore(url, embedder.dimensions)
    store.create_schema()
    for strategy in ("A", "B"):
        data = chunks[strategy]
        store.upsert(strategy, data, embedder.embed_documents([c.text for c in data]))
    ingest_seconds = round(time.perf_counter() - started, 2)

    traces = []
    for q in questions:
        vector = embedder.embed_query(q["question"])
        for strategy in ("A", "B", "C"):
            filters = q.get("filter") if strategy == "C" else None
            hits = store.search("A" if strategy == "A" else "B", vector, top_k=3, filters=filters)
            trace = evaluate_query(q, hits)
            trace.update({"strategy": strategy, "filter_applied": filters,
                          "retrieval_strategy_in_database": "A" if strategy == "A" else "B",
                          "filter_compliant": (all(str(h["metadata"].get(k)) == str(v)
                                                   for h in hits for k, v in filters.items())
                                               if filters and hits else None)})
            traces.append(trace)

    metrics = summarize(traces)
    metrics["_notes"] = {
        "unmatched_gold_quote_question_ids": mismatches,
        "accuracy": "Exact quote-and-location proxy; manually review other valid evidence",
        "coherence": "Human scores required (0-2)",
        "grounding": "Human scores required (0-2)",
        "strategy_C": "Shares B chunks and vectors; applies metadata filters when supplied",
        "model_name": embedder.model_name, "dimensions": embedder.dimensions,
        "ingest_and_embed_seconds": ingest_seconds,
        "chunks": {s: len(chunks[s]) for s in ("A", "B", "C")},
        "chunk_mean_length": {s: round(statistics.mean(map(lambda c: len(c.text), chunks[s])), 2)
                              for s in ("A", "B")},
        "chunk_total_characters": {s: sum(len(c.text) for c in chunks[s]) for s in ("A", "B")},
        "api_fees_usd": 0, "local_compute_energy_cost": "not measured",
        "gold_quote_found_in_source_page": quotes, "corpus": corpus,
        "filter_compliance_by_strategy": {
            s: {"filter_queries": len(group),
                "all_top3_match_filter": sum(t["filter_compliant"] is True for t in group),
                "empty_searches": sum(not t["hits"] for t in group)}
            for s in ("A", "B", "C")
            for group in [[t for t in traces if t["strategy"] == s and t["filter_applied"]]]
        },
        "metadata_presence_by_strategy": {
            s: {"top3_chunks_with_source_page_document_type": sum(
                    bool(h.get("source")) and isinstance(h.get("page"), int)
                    and bool(h.get("metadata", {}).get("document_type"))
                    for t in traces if t["strategy"] == s and t["split"] == "test" for h in t["hits"]),
                "test_top3_chunks_returned": sum(len(t["hits"]) for t in traces
                                                if t["strategy"] == s and t["split"] == "test")}
            for s in ("A", "B", "C")
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "retrieval_traces.json").write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "retrieval_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    save_reviews(pages, chunks, traces)
    for strategy in ("A", "B", "C"):
        for split in ("dev", "test"):
            result = metrics[strategy][split]
            print(f"{strategy} {split}: Hit@1={result['hit_at_1']['percentage']}%, "
                  f"Evidence Hit@3={result['evidence_hit_at_3']['percentage']}%")
    print("Saved retrieval traces, metrics and manual review CSVs to results.")


if __name__ == "__main__":
    main()
