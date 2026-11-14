"""PostgreSQL + pgvector smoke test.
"""
import os
import sys
from pathlib import Path

# Allow running this file directly with: python scripts/test_store.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg
from dotenv import load_dotenv

from src.models import Chunk
from src.store import PgVectorStore


def main() -> None:
    starter = Path(__file__).resolve().parents[1]
    load_dotenv(starter / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Missing DATABASE_URL in starter/.env")

    store = PgVectorStore(database_url, dimensions=384)
    store.create_schema()
    strategy = "_day6_store_smoke_test"
    doc_id = "_day6_test_document"

    def chunk(index: int, text: str, doc_type: str = "lecture_slides") -> Chunk:
        return Chunk(
            chunk_id=f"{doc_id}_p001_test_{index:03d}",
            document_id=doc_id,
            source="smoke_test.pdf",
            page=1,
            text=text,
            metadata={"document_type": doc_type, "page_id": f"{doc_id}_p001"},
        )

    def count_rows() -> int:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM day6_chunks WHERE strategy = %s", (strategy,))
                return cur.fetchone()[0]

    try:
        docs = [chunk(1, "Supervised learning uses labeled data."),
                chunk(2, "Unsupervised learning finds patterns.", "book_chapter")]
        vectors = [[1.0] + [0.0] * 383, [0.0, 1.0] + [0.0] * 382]
        store.upsert(strategy, docs, vectors)
        assert count_rows() == 2, "Expected 2 rows"

        store.upsert(strategy, docs, vectors)
        assert count_rows() == 2, "Duplicate rows after re-ingestion"

        results = store.search(strategy, vectors[0], top_k=2)
        assert len(results) == 2
        assert results[0]["chunk_id"] == docs[0].chunk_id
        assert results[0]["score"] > results[1]["score"]
        assert results[0]["metadata"]["document_type"] == "lecture_slides"

        filtered = store.search(
            strategy, vectors[0], top_k=3,
            filters={"document_type": "book_chapter"}
        )
        assert len(filtered) == 1 and filtered[0]["chunk_id"] == docs[1].chunk_id
        assert store.search(strategy, vectors[0], filters={"document_id": doc_id})

        store.upsert(strategy, docs[:1], vectors[:1])
        assert count_rows() == 1, "Obsolete chunk was not removed"
        print("Schema creation: PASSED")
        print("Idempotent upsert: PASSED")
        print("Cosine search: PASSED")
        print("Metadata filtering: PASSED")
        print("Stale chunk cleanup: PASSED")
        print("Store test PASSED")
    finally:
        with psycopg.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM day6_chunks WHERE strategy = %s", (strategy,))


if __name__ == "__main__":
    main()
