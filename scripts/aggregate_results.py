"""Aggregate benchmark results into the numbers used in the README."""

import json
import statistics
from pathlib import Path

RESULTS_PATH = Path("evaluation_results.json")


def summarize(rows: list[dict]) -> dict:
    """Mean scores and median latency for one group of rows."""
    return {
        "n": len(rows),
        "faithfulness": statistics.mean(r["faithfulness"] for r in rows),
        "context_precision": statistics.mean(r["context_precision"] for r in rows),
        "latency_ms": statistics.median(r["latency_ms"] for r in rows),
    }


def report(label: str, rows: list[dict]) -> None:
    if not rows:
        return
    s = summarize(rows)
    print(
        f"  {label:<12} n={s['n']:<3} "
        f"faith={s['faithfulness']:.2f}  "
        f"ctx_prec={s['context_precision']:.2f}  "
        f"lat_median={s['latency_ms']:.0f}ms"
    )


def main() -> None:
    rows = json.loads(RESULTS_PATH.read_text())

    for backend in ["custom", "bedrock"]:
        backend_rows = [r for r in rows if r["backend"] == backend]
        print(f"\n{backend.upper()}")
        report("overall", backend_rows)

        for qtype in ["synthesis", "specific"]:
            report(qtype, [r for r in backend_rows if r["question_type"] == qtype])

        for ticker in sorted({r["ticker"] for r in backend_rows}):
            report(ticker, [r for r in backend_rows if r["ticker"] == ticker])


if __name__ == "__main__":
    main()