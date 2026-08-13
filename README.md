## Retrieval approaches: custom pipeline vs. AWS Bedrock Knowledge Base

Same retrieval task, two implementations behind one interface, benchmarked on
identical questions.

### Method

6 questions across AAPL, NVDA and TSLA. Half are broad synthesis questions,
half name specific programs or entities. Each ran through both backends.
Quality scored with RAGAS faithfulness and context precision (both
reference-free). Latency measured on the retriever functions alone, median of 3
runs after a warm-up, so the shared yFinance and GPT-4o steps don't drown out
the difference I'm trying to measure.

Raw output in `evaluation_results.json`. Reproduce with
`python -m scripts.run_comparison` then `python -m scripts.aggregate_results`.

### Results

| Dimension | Custom | Bedrock KB |
|---|---|---|
| Faithfulness | 0.99 | 0.97 |
| Context precision | 0.83 | 0.97 |
| — synthesis questions | 0.94 | 0.97 |
| — specific-term questions | 0.71 | 0.97 |
| Median retrieval latency | 360 ms | 819 ms |
| Cost model | per-query embedding + self-hosted store | per-GB indexed + per-query |
| Retrieval | hybrid (vector + BM25), tunable weights | hybrid, managed, not tunable |
| Reranking | cross-encoder, swappable | managed, model undisclosed |
| Metadata filtering | Chroma `where`, set at ingest | S3 `.metadata.json`, needs re-sync |
| Ops | mine to maintain | AWS's |

### What the numbers show

I expected BM25 to give my pipeline the edge on specific-term questions. It lost
there by the widest margin in the benchmark, 0.71 to 0.97. Faithfulness was a
wash, so context precision is the only place the two really separate.

Some of that gap is chunk size, not retrieval quality. My chunks are ~1000
chars; Bedrock's are 1500-2500. Context precision asks a judge whether each
chunk is relevant, and bigger chunks clear that bar more easily.

The rest is my reranker. Context precision is rank-aware, and my worst score
(AAPL specific-term, 0.50) came from the cross-encoder putting a boilerplate
stock-price passage above the Shareholders' Equity note that actually had the
buyback numbers. Hybrid search found the right text. The reranker buried it.
Larger chunks and a better reranker are the next thing to try, measured against
this same harness.

Worth noting in the other direction: Bedrock's lowest faithfulness score (0.84,
TSLA) is a real error the metric caught. That answer presents deferred-revenue
figures as regulatory-credit revenue. The retrieved chunk says otherwise.

### Caveats

Latency isn't apples to apples. The custom path makes an OpenAI embedding call,
searches local Chroma, then reranks on CPU. Bedrock is one API call to
us-east-1. Both hit the network, just differently. This measures my deployment,
not the approaches in general.

RAGAS metrics here are reference-free LLM-as-judge, roughly 80% agreement with
human raters. Fine for comparing two systems, not ground truth.

6 questions is a small sample. Directional only.

### When I'd choose each

**Custom** when I need to tune retrieval myself (chunk size, hybrid weights,
reranker), when cloud lock-in is a concern, or when per-query cost at volume
beats the ops burden. The benchmark is the argument for control: it told me
exactly which component to fix, which only helps because I own that component.

**Bedrock** for a team already on AWS that wants no retrieval infra to maintain
and can live with an opaque strategy. It beat a hand-tuned pipeline here with
zero tuning from me.

### Portability

Both retrievers are `(query, ticker, top_n) -> list[str]`, so the agent picks
between them with one branch. A third backend (Azure AI Search, Vertex AI
Search) means one new module against that signature and nothing else changes.