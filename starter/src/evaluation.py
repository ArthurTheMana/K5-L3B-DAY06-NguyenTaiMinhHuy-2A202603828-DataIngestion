"""TODO (starter): evaluate retrieved evidence and preserve per-query traces.
"""

import re
import unicodedata
from collections import defaultdict
from typing import Any


def _clean(text: str) -> str:
    """Normalize text before comparing evidence."""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _gold_locations(question: dict[str, Any]) -> set[tuple[str, int]]:
    """Collect original and alternative gold evidence locations."""

    locations = {
        (str(question["gold_source_id"]), int(question["gold_page"]))
    }

    for alternative in question.get("alternative_gold", []):
        locations.add((
            str(alternative["gold_source_id"]),
            int(alternative["gold_page"])
        ))

    return locations


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]]
) -> dict[str, Any]:
    """TODO (starter):
    Check source/page and enough gold evidence.
    Keep Hit@1 and Evidence Hit@3 traces.
    Accept documented alternative gold evidence.
    Do not infer human grounding or chunk coherence.
    """

    gold = _gold_locations(question)

    quotes = [question.get("evidence_quote", "")]
    quotes += list(question.get("alternative_evidence_quotes", []))

    normalized_quotes = [
        _clean(q) for q in quotes if _clean(q)
    ]

    reviewed_hits = []

    for rank, hit in enumerate(hits[:3], start=1):

        location = (
            str(hit.get("document_id")),
            int(hit.get("page", -1))
        )

        at_gold_location = location in gold
        content = _clean(hit.get("text", ""))

        evidence_found = bool(normalized_quotes) and any(
            quote in content for quote in normalized_quotes
        )

        exact_hit = at_gold_location and evidence_found

        reviewed_hits.append({
            **hit,
            "rank": rank,
            "gold_location_match": at_gold_location,
            "exact_evidence_match": evidence_found,
            "auto_evidence_hit": exact_hit,
            "manual_evidence_override": None
        })

    hit_at_1 = int(
        bool(reviewed_hits and reviewed_hits[0]["auto_evidence_hit"])
    )

    evidence_hit_at_3 = int(
        any(hit["auto_evidence_hit"] for hit in reviewed_hits)
    )

    location_hit_at_3 = int(
        any(hit["gold_location_match"] for hit in reviewed_hits)
    )

    requires_manual_review = any(
        hit["gold_location_match"] and not hit["auto_evidence_hit"]
        for hit in reviewed_hits
    )

    return {
        "id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold_source_id": question["gold_source_id"],
        "gold_page": question["gold_page"],
        "evidence_quote": question["evidence_quote"],
        "filter": question.get("filter"),
        "hits": reviewed_hits,
        "hit_at_1": hit_at_1,
        "evidence_hit_at_3": evidence_hit_at_3,
        "location_hit_at_3": location_hit_at_3,
        "requires_manual_review": bool(requires_manual_review)
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """TODO (starter):
    Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split.
    """

    grouped = defaultdict(list)

    for trace in traces:
        key = (trace["strategy"], trace["split"])
        grouped[key].append(trace)

    results = {}

    for (strategy, split), group in sorted(grouped.items()):

        total = len(group)

        hit1 = sum(
            int(trace["hit_at_1"]) for trace in group
        )

        hit3 = sum(
            int(trace["evidence_hit_at_3"]) for trace in group
        )

        manual_review_count = sum(
            bool(trace["requires_manual_review"])
            for trace in group
        )

        results.setdefault(strategy, {})[split] = {
            "questions": total,
            "hit_at_1": {
                "hits": hit1,
                "total": total,
                "percentage": round(hit1 * 100 / total, 2)
                if total else None
            },
            "evidence_hit_at_3": {
                "hits": hit3,
                "total": total,
                "percentage": round(hit3 * 100 / total, 2)
                if total else None
            },
            "needs_manual_evidence_review": manual_review_count
        }

    return results
