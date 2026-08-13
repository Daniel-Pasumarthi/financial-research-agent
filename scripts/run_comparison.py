"""Benchmark the custom retrieval pipeline against AWS Bedrock Knowledge Base."""

import argparse
import json
import statistics
import time
from pathlib import Path

from app.agent import build_agent
from app.evaluation import evaluate_run
from app.retriever_bedrock import retrieve_bedrock
from app.retriever_custom import retrieve_custom

LATENCY_RUNS = 3
TOP_N = 4
BACKENDS = ["custom", "bedrock"]

RETRIEVERS = {
    "custom": retrieve_custom,
    "bedrock": retrieve_bedrock,
}

QUESTIONS = [
    {"ticker": "AAPL", "type": "synthesis",
     "query": "What are the main risks to the company's revenue growth?"},
    {"ticker": "AAPL", "type": "specific",
     "query": "What does the company say about its share repurchase program and dividends?"},
    {"ticker": "NVDA", "type": "synthesis",
     "query": "How does the company describe its competitive position in the market?"},
    {"ticker": "NVDA", "type": "specific",
     "query": "What does the company say about export restrictions affecting sales to China?"},
    {"ticker": "TSLA", "type": "synthesis",
     "query": "What are the key risks facing the company's business?"},
    {"ticker": "TSLA", "type": "specific",
     "query": "What does the company say about revenue from regulatory credits?"},
]


def median_retrieval_latency(backend: str, query: str, ticker: str) -> float:
    """Time a retriever in isolation. Returns the median of LATENCY_RUNS, in ms."""
    retriever = RETRIEVERS[backend]

    retriever(query, ticker, top_n=TOP_N)  # warm-up, deliberately not timed

    timings = []
    for _ in range(LATENCY_RUNS):
        start = time.perf_counter()
        retriever(query, ticker, top_n=TOP_N)
        timings.append((time.perf_counter() - start) * 1000)

    return statistics.median(timings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first question, to a separate output file.",
    )
    args = parser.parse_args()

    questions = QUESTIONS[:1] if args.smoke else QUESTIONS
    output_path = Path(
        "evaluation_results.smoke.json" if args.smoke else "evaluation_results.json"
    )

    agent = build_agent()
    rows = []

    for item in questions:
        for backend in BACKENDS:
            print(f"Running {item['ticker']} / {item['type']} / {backend} ...")

            latency_ms = median_retrieval_latency(
                backend, item["query"], item["ticker"]
            )

            result = agent.invoke({
                "ticker": item["ticker"],
                "query": item["query"],
                "retrieval_backend": backend,
                "context": [],
                "financials": {},
                "answer": "",
            })

            scores = evaluate_run(
                query=item["query"],
                answer=result["answer"],
                contexts=result["context"],
            )

            rows.append({
                "ticker": item["ticker"],
                "question_type": item["type"],
                "query": item["query"],
                "backend": backend,
                "latency_ms": round(latency_ms, 1),
                "faithfulness": scores["faithfulness"],
                "context_precision": scores["context_precision"],
                "answer": result["answer"],
                "contexts": result["context"],
            })

            output_path.write_text(json.dumps(rows, indent=2, default=str))
            print(
                f"  faithfulness={scores['faithfulness']:.3f}  "
                f"context_precision={scores['context_precision']:.3f}  "
                f"latency={latency_ms:.0f}ms"
            )

    print(f"\nWrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()