# Financial Research Agent

A stateful **LangGraph agent** that answers financial research questions grounded in SEC 10-K filings via RAG, with multi-ticker support (currently the MAG-7) and earnings lookups via yFinance.

## How it's built

- **Two-stage hybrid retrieval** — vector + BM25 search with cross-encoder reranking, improving context precision over naive vector search (measured with RAGAS)
- **Dual-backend design** — a custom retrieval pipeline and an AWS Bedrock Knowledge Base sit behind one interface, switchable with a one-line toggle. I benchmarked both on latency, cost, and RAGAS quality scores instead of just picking one — the kind of tradeoff call that matters in a real production system.
- **Stateful LangGraph orchestration** — multi-step reasoning and tool calls (filing retrieval, earnings lookups) run as a graph, not a single prompt chain
- **Metadata-filtered retrieval** so queries stay scoped to the right company and filing

## Stack

`Python` · `LangGraph` · `ChromaDB` · `AWS Bedrock Knowledge Base` · `yFinance` · `Streamlit` · `Docker` · `LangSmith` · `RAGAS`

## Running locally

```bash
git clone https://github.com/Daniel-Pasumarthi/financial-research-agent.git
cd financial-research-agent
cp .env.example .env   # add your API keys
docker-compose up --build
```

Retrieval quality and agent responses are evaluated on-demand with **RAGAS**, fully traced via **LangSmith** — every benchmark result is reproducible, not a one-off number.
