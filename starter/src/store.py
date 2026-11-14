"""TODO (starter): store and search vectors using PostgreSQL + pgvector.

create schema, idempotent upserts, stale-chunk cleanup, cosine search,
and safe SQL metadata filtering. Requires psycopg[binary] and pgvector extension.
"""
import math
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from .models import Chunk


# Only explicitly supported user-controlled filter names are permitted.
# 'document_type' is needed by the Day 6 20-question benchmark.
_COLUMN_FILTERS = frozenset({"document_id", "source", "page"})
_METADATA_FILTERS = frozenset({
    "document_type", "source_id", "year", "section", "category",
    "audience", "language", "version",
})


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        """TODO (starter): retain connection settings; validate model dimensions.

        """
        if not isinstance(database_url, str) or not database_url.strip():
            raise ValueError("database_url must be a non-empty string")
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        if dimensions > 16000:
            raise ValueError("pgvector vector dimensions cannot exceed 16000")
        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        """TODO (starter): create table with vector column and source/page/filter metadata.
        """
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(sql.SQL("""
                    CREATE TABLE IF NOT EXISTS day6_chunks (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding VECTOR({}) NOT NULL,
                        PRIMARY KEY (strategy, chunk_id)
                    )
                """).format(sql.SQL(str(self.dimensions))))
                # CREATE TABLE IF NOT EXISTS will not fix an old dimension.
                cur.execute("""
                    SELECT a.atttypmod
                    FROM pg_attribute AS a
                    WHERE a.attrelid = 'day6_chunks'::regclass
                      AND a.attname = 'embedding'
                      AND NOT a.attisdropped
                """)
                actual_dimensions = cur.fetchone()[0]
                if actual_dimensions != self.dimensions:
                    raise ValueError(
                        f"Existing day6_chunks vector dimension is {actual_dimensions}; "
                        f"requested {self.dimensions}. Use the same embedding model."
                    )
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS day6_chunks_strategy_document_idx
                    ON day6_chunks (strategy, document_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS day6_chunks_strategy_source_page_idx
                    ON day6_chunks (strategy, source, page)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS day6_chunks_metadata_idx
                    ON day6_chunks USING GIN (metadata)
                """)
        # Note: exact KNN sorting is adequate for a small Day 6 corpus.

    def _vector_literal(self, vector: list[float]) -> str:
        """Validate a vector and serialize it for a parameterized ::vector cast."""
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Expected vector of {self.dimensions} dimensions, got {len(vector)}"
            )
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
            raise ValueError("Vector values must be finite numbers")
        return "[" + ",".join(str(float(x)) for x in vector) + "]"

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """TODO (starter): validate lengths/dimensions and write stable IDs without duplicates.
        TODO (starter): when re-ingesting changed source documents, remove obsolete chunks too.
        """
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("strategy must be a non-empty string")
        if len(chunks) != len(vectors):
            raise ValueError("The number of chunks must match the number of vectors")
        if not chunks:
            return  # Cannot infer which document to clear from an empty input.

        ids: set[str] = set()
        ids_by_document: dict[str, list[str]] = {}
        records = []
        for chunk, vector in zip(chunks, vectors):
            if not chunk.chunk_id or not chunk.document_id:
                raise ValueError("Every chunk needs a stable chunk_id and document_id")
            if chunk.chunk_id in ids:
                raise ValueError(f"Duplicate chunk_id in input: {chunk.chunk_id}")
            ids.add(chunk.chunk_id)
            ids_by_document.setdefault(chunk.document_id, []).append(chunk.chunk_id)
            records.append((
                strategy, chunk.chunk_id, chunk.document_id, chunk.source,
                chunk.page, chunk.text, Jsonb(chunk.metadata),
                self._vector_literal(vector),
            ))

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO day6_chunks
                        (strategy, chunk_id, document_id, source, page, text, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        source = EXCLUDED.source,
                        page = EXCLUDED.page,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                """, records)
                for document_id, current_ids in ids_by_document.items():
                    cur.execute("""
                        DELETE FROM day6_chunks
                        WHERE strategy = %s AND document_id = %s
                          AND NOT (chunk_id = ANY(%s::text[]))
                    """, (strategy, document_id, current_ids))

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """TODO (starter): filter permitted metadata in SQL, rank by cosine distance, return top-k.
        TODO (starter): return chunk ID, text, score and metadata; parameterize values.
        """
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("strategy must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        vector_literal = self._vector_literal(query_vector)
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("filters must be a dictionary or None")

        clauses = [sql.SQL("strategy = %s")]
        params: list[Any] = [strategy]
        for key, value in (filters or {}).items():
            if key in _COLUMN_FILTERS:
                if key == "page":
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise ValueError("page filter must be an integer")
                elif not isinstance(value, str):
                    raise ValueError(f"{key} filter must be a string")
                clauses.append(sql.SQL("{} = %s").format(sql.Identifier(key)))
                params.append(value)
            elif key in _METADATA_FILTERS:
                if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                    raise ValueError(f"{key} filter must be a string or number")
                clauses.append(sql.SQL("metadata ->> %s = %s"))
                params.extend([key, str(value)])
            else:
                raise ValueError(f"Unsupported filter key: {key!r}")

        query = sql.SQL("""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   (1 - (embedding <=> %s::vector)) AS score
            FROM day6_chunks
            WHERE {}
            ORDER BY embedding <=> %s::vector, chunk_id
            LIMIT %s
        """).format(sql.SQL(" AND ").join(clauses))
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, [vector_literal, *params, vector_literal, top_k])
                rows = cur.fetchall()
        return [
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source": source,
                "page": page,
                "text": text,
                "score": float(score),
                "metadata": metadata,
            }
            for chunk_id, document_id, source, page, text, metadata, score in rows
        ]
