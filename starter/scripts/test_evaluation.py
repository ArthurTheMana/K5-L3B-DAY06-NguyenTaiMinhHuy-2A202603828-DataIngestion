"""Smoke test: Evaluation test to see if the answer are all good.
"""
from pathlib import Path
import sys

STARTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STARTER))

from src.evaluation import evaluate_query, summarize


def question(qid="Q01", split="dev"):
    return {
        "id": qid,
        "split": split,
        "question": "What is supervised learning?",
        "expected_answer": "Learning from labeled examples.",
        "gold_source_id": "uow_ai_ml",
        "gold_page": 12,
        "evidence_quote": "learning from labeled examples",
        "filter": None,
    }


def hit(document_id="uow_ai_ml", page=12, text="Supervised learning means learning from labeled examples."):
    return {
        "chunk_id": f"{document_id}_p{page}",
        "document_id": document_id,
        "source": "test.pdf",
        "page": page,
        "text": text,
        "score": 0.8,
        "metadata": {},
    }


def main():
    # 1. Exact evidence at rank 1: both metrics are 1.
    first = evaluate_query(question("Q01"), [hit()])
    assert first["hit_at_1"] == 1
    assert first["evidence_hit_at_3"] == 1
    assert first["hits"][0]["rank"] == 1
    print("Top-1 evidence: PASSED")

    # 2. Evidence only at rank 2: Hit@1=0, Evidence Hit@3=1.
    second = evaluate_query(question("Q02"), [hit("other", 3, "irrelevant"), hit()])
    assert second["hit_at_1"] == 0
    assert second["evidence_hit_at_3"] == 1
    assert [h["rank"] for h in second["hits"]] == [1, 2]
    print("Top-3 evidence: PASSED")

    # 3. Matching text from the wrong source or page must not count.
    wrong = evaluate_query(question("Q03"), [hit("other", 12), hit("uow_ai_ml", 99)])
    assert wrong["hit_at_1"] == 0
    assert wrong["evidence_hit_at_3"] == 0
    assert wrong["location_hit_at_3"] == 0
    print("Source/page validation: PASSED")

    # 4. Same source/page but quote missing needs human review.
    uncertain = evaluate_query(question("Q04"), [hit(text="A discussion of supervised learning with different wording.")])
    assert uncertain["hit_at_1"] == 0
    assert uncertain["location_hit_at_3"] == 1
    assert uncertain["requires_manual_review"] is True
    print("Manual review flag: PASSED")

    # 5. Explicit alternative gold location/quote is accepted.
    alternative = question("Q05", "test")
    alternative["alternative_gold"] = [{"gold_source_id": "colliot_ml_2023", "gold_page": 4}]
    alternative["alternative_evidence_quotes"] = ["training on examples with known labels"]
    alternative_result = evaluate_query(alternative, [hit("colliot_ml_2023", 4, "Training on examples with known labels.")])
    assert alternative_result["hit_at_1"] == 1
    print("Alternative gold evidence: PASSED")

    # 6. Summaries must separate strategies and dev/test splits.
    traces = [
        dict(first, strategy="A"),
        dict(second, strategy="A"),
        dict(wrong, strategy="B"),
        dict(alternative_result, strategy="C"),
    ]
    report = summarize(traces)
    assert report["A"]["dev"]["questions"] == 2
    assert report["A"]["dev"]["hit_at_1"]["hits"] == 1
    assert report["A"]["dev"]["hit_at_1"]["percentage"] == 50.0
    assert report["A"]["dev"]["evidence_hit_at_3"]["percentage"] == 100.0
    assert report["B"]["dev"]["hit_at_1"]["percentage"] == 0.0
    assert report["C"]["test"]["hit_at_1"]["percentage"] == 100.0
    print("Strategy / dev-test summary: PASSED")

    # 7. At most top-3 should be counted, even when rank 4 has evidence.
    beyond = evaluate_query(question("Q06"), [hit("other", 1, "no"), hit("other", 2, "no"), hit("other", 3, "no"), hit()])
    assert len(beyond["hits"]) == 3
    assert beyond["evidence_hit_at_3"] == 0
    print("Top-3 limit: PASSED")

    print("\nEvaluation test PASSED")


if __name__ == "__main__":
    main()
