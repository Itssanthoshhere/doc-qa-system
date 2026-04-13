#!/usr/bin/env python3
"""
scripts/evaluate.py — Offline evaluation on SQuAD 2.0 or custom QA datasets.

Measures:
  - Exact Match (EM)
  - F1 Score
  - Retrieval recall@k
  - Latency percentiles (p50, p90, p99)

Usage:
  python scripts/evaluate.py \
    --dataset data/squad_dev.json \
    --doc_dir data/eval_docs/ \
    --output data/eval_results.json
"""
import argparse
import json
import time
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.pipeline import DocSagePipeline
from core.logging import configure_logging


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation and articles — standard SQuAD normalization."""
    import re, string
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = sum(
        min(pred_tokens.count(t), truth_tokens.count(t))
        for t in set(truth_tokens)
    )
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def evaluate_dataset(
    dataset_path: Path,
    pipeline: DocSagePipeline,
    session_id: str,
) -> dict[str, Any]:
    with open(dataset_path) as f:
        dataset = json.load(f)

    results = []
    latencies = []
    f1_scores = []
    em_scores = []

    questions = dataset if isinstance(dataset, list) else dataset.get("data", [])

    for item in questions:
        question = item["question"]
        ground_truths = item.get("answers", [])
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        if not ground_truths:
            continue

        t_start = time.perf_counter()
        response = pipeline.answer(question=question, session_id=session_id)
        latency_ms = (time.perf_counter() - t_start) * 1000
        latencies.append(latency_ms)

        prediction = response.answer
        item_f1 = max(compute_f1(prediction, gt) for gt in ground_truths)
        item_em = any(compute_em(prediction, gt) for gt in ground_truths)
        f1_scores.append(item_f1)
        em_scores.append(float(item_em))

        results.append({
            "question": question,
            "prediction": prediction,
            "ground_truth": ground_truths[0],
            "f1": item_f1,
            "em": item_em,
            "confidence": response.confidence,
            "latency_ms": latency_ms,
            "adversarial_risk": response.adversarial_risk,
        })

    latencies.sort()
    n = len(latencies)

    summary = {
        "num_questions": len(results),
        "f1_mean": round(sum(f1_scores) / len(f1_scores) * 100, 2),
        "em_mean": round(sum(em_scores) / len(em_scores) * 100, 2),
        "latency_p50_ms": round(latencies[int(n * 0.50)], 1) if latencies else 0,
        "latency_p90_ms": round(latencies[int(n * 0.90)], 1) if latencies else 0,
        "latency_p99_ms": round(latencies[min(int(n * 0.99), n - 1)], 1) if latencies else 0,
        "confidence_mean": round(
            sum(r["confidence"] for r in results) / len(results), 4
        ) if results else 0,
    }

    return {"summary": summary, "results": results}


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="Evaluate DocSage QA pipeline")
    parser.add_argument("--dataset", required=True, help="Path to QA dataset JSON")
    parser.add_argument("--output", default="eval_results.json", help="Output path")
    parser.add_argument("--ingest_dir", default=None, help="Directory of docs to ingest first")
    args = parser.parse_args()

    pipeline = DocSagePipeline.get()
    session_id = pipeline.create_session()

    # Optionally ingest evaluation documents
    if args.ingest_dir:
        doc_dir = Path(args.ingest_dir)
        for doc_path in sorted(doc_dir.glob("*")):
            if doc_path.is_file():
                print(f"Ingesting {doc_path.name}...")
                pipeline.ingest_document(doc_path)

    print(f"Evaluating on {args.dataset}...")
    eval_results = evaluate_dataset(Path(args.dataset), pipeline, session_id)

    summary = eval_results["summary"]
    print("\n── Results ──────────────────────────────")
    print(f"  Questions evaluated : {summary['num_questions']}")
    print(f"  F1 Score            : {summary['f1_mean']}%")
    print(f"  Exact Match         : {summary['em_mean']}%")
    print(f"  Latency p50         : {summary['latency_p50_ms']} ms")
    print(f"  Latency p90         : {summary['latency_p90_ms']} ms")
    print(f"  Latency p99         : {summary['latency_p99_ms']} ms")
    print(f"  Mean confidence     : {summary['confidence_mean']}")
    print("─────────────────────────────────────────\n")

    with open(args.output, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Full results written to {args.output}")


if __name__ == "__main__":
    main()