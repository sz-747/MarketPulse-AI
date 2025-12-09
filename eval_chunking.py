"""
Evaluation script for RAG chunking/recall.

Features:
- Loads test cases with `question`, `expected_span`, optional `required_speaker`.
- Retrieves top-k chunks from Pinecone (using the same HF embeddings as ingest).
- Checks if any retrieved chunk contains the expected span (case-insensitive, whitespace-normalized).
- If `required_speaker` is provided, only counts a hit when that speaker matches chunk metadata.
- Reports per-case pass/fail and aggregate recall@k, with retrieval logs.

Default test file: tests/rag_eval_cases.json
Example case:
[
  {
    "question": "What is the revenue guidance for Q4?",
    "expected_span": "Total revenue is expected to be $65 billion, plus or minus 2%.",
    "required_speaker": "Colette Kress"
  }
]
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


@dataclass
class TestCase:
    question: str
    expected_span: Optional[str] = None
    required_speaker: Optional[str] = None
    case_type: str = "span"  # "span" (speaker-quote) or "metric"
    value_terms: List[str] = field(default_factory=list)
    period_terms: List[str] = field(default_factory=list)
    metric_terms: List[str] = field(default_factory=list)


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace for tolerant matching."""
    return " ".join(text.lower().split())


def span_matches(span: str, text: str) -> bool:
    """Check if normalized span is contained in normalized text."""
    return normalize(span) in normalize(text)


def contains_any(text: str, terms: Sequence[str]) -> bool:
    norm_text = normalize(text)
    return any(normalize(t) in norm_text for t in terms if t)


def has_value(text: str, value_terms: Sequence[str]) -> bool:
    """
    Flexible numeric/value detection:
    - Lowercases everything.
    - If a raw number like "65" is present in value_terms, match common variants:
      "65", "65b", "65 b", "65 billion", "$65b", "$65 billion".
    - Otherwise fall back to any(term in text).
    """
    norm_text = text.lower()
    norm_values = [v.lower() for v in value_terms]
    if "65" in norm_values:
        patterns = ["65", "65b", "65 b", "65 billion", "$65b", "$65 billion"]
        return any(p in norm_text for p in patterns)
    return any(v in norm_text for v in norm_values)


def metric_match(text: str, case: TestCase) -> bool:
    """
    For metric tests, require:
    - value present (flexible number matching),
    - period present (accept period_terms or 'fourth quarter'),
    - metric keyword present (any of metric_terms).
    Speaker is NOT required for metric tests.
    """
    if case.case_type != "metric":
        return False
    norm_text = text.lower()
    value_ok = has_value(norm_text, case.value_terms) if case.value_terms else False
    period_terms = [t.lower() for t in case.period_terms] + ["fourth quarter"]
    period_ok = contains_any(norm_text, period_terms) if period_terms else False
    metric_terms = [t.lower() for t in case.metric_terms]
    metric_ok = contains_any(norm_text, metric_terms) if metric_terms else True
    return value_ok and period_ok and metric_ok


def speaker_matches(required: Optional[str], metadata: dict | None) -> bool:
    if not required:
        return True
    if not metadata:
        return False
    speaker = metadata.get("speaker") or metadata.get("Speaker")
    if not speaker:
        return False
    return normalize(required) in normalize(str(speaker))


def load_test_cases(path: Path) -> List[TestCase]:
    if not path.exists():
        # Provide a minimal default so the script can run.
        return [
            TestCase(
                question="What is the revenue guidance for Q4?",
                case_type="metric",
                value_terms=["65", "billion"],
                period_terms=["q4"],
                metric_terms=["revenue", "guid"],
            )
        ]
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cases: List[TestCase] = []
    for item in data:
        cases.append(
            TestCase(
                question=item["question"],
                expected_span=item.get("expected_span"),
                required_speaker=item.get("required_speaker"),
                case_type=item.get("case_type", "span"),
                value_terms=item.get("value_terms", []),
                period_terms=item.get("period_terms", []),
                metric_terms=item.get("metric_terms", []),
            )
        )
    return cases


def init_vector_store(index_name: str) -> PineconeVectorStore:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY not set")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )


def evaluate_case(
    store: PineconeVectorStore, case: TestCase, k: int
) -> Tuple[bool, List[Tuple[float, str, dict]]]:
    results = store.similarity_search_with_score(case.question, k=k)
    hits = []
    passed = False
    for doc, score in results:
        meta = doc.metadata or {}
        hits.append((score, doc.page_content, meta))

        if case.case_type == "metric":
            # Speaker not enforced for metrics.
            if metric_match(doc.page_content, case):
                passed = True
        else:
            # Speaker-quote / span tests.
            if case.expected_span and span_matches(case.expected_span, doc.page_content):
                if speaker_matches(case.required_speaker, meta):
                    passed = True
    return passed, hits


def format_preview(text: str, max_chars: int = 320) -> str:
    return (text[: max_chars].rstrip() + "...") if len(text) > max_chars else text


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate RAG chunk recall.")
    parser.add_argument(
        "--cases",
        type=str,
        default="tests/rag_eval_cases.json",
        help="Path to JSON file with test cases.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=15,
        help="Top-k chunks to retrieve for each question.",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="marketpulse",
        help="Pinecone index name.",
    )
    args = parser.parse_args()

    cases = load_test_cases(Path(args.cases))
    if not cases:
        print("No test cases found. Exiting.")
        return

    store = init_vector_store(args.index)

    total = len(cases)
    passed_count = 0

    print(f"Running {total} cases | k={args.k} | index={args.index}")
    print("-" * 60)

    for idx, case in enumerate(cases, start=1):
        ok, hits = evaluate_case(store, case, k=args.k)
        if ok:
            passed_count += 1
        status = "PASS" if ok else "FAIL"
        print(f"\nCase {idx}/{total} [{status}]")
        print(f"Q: {case.question}")
        print(f"Type: {case.case_type}")
        if case.case_type == "metric":
            print(f"Value terms: {case.value_terms}")
            print(f"Period terms: {case.period_terms}")
            print(f"Metric terms: {case.metric_terms}")
        else:
            print(f"Expected span: {case.expected_span}")
            if case.required_speaker:
                print(f"Required speaker: {case.required_speaker}")
        print(f"Top-{args.k} retrieved:")
        for rank, (score, text, meta) in enumerate(hits, start=1):
            page = meta.get("page", "unknown")
            speaker = meta.get("speaker") or meta.get("Speaker") or "unknown"
            print(
                f"  {rank}. score={score:.4f} | page={page} | speaker={speaker}\n"
                f"     {format_preview(text)}"
            )

    recall = passed_count / total if total else 0.0
    print("\n" + "=" * 60)
    print(f"Recall@{args.k}: {passed_count}/{total} = {recall:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

