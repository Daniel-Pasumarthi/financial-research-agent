# Financial Research Agent

A stateful **LangGraph agent** that retrieves grounded context from SEC 10-K filings via RAG and surfaces earnings insights using yFinance — with a dual-backend retrieval architecture you can toggle with one line.

## What it does

This agent answers financial research questions grounded in real SEC 10-K filings, with multi-ticker support (currently the MAG-7). It's built to show — not just claim — that retrieval quality matters, by making the retrieval backend a swappable, benchmarkable component rather than a fixed choice.

## Architecture

- **Two-stage retrieval pipeline** — hybrid vector + BM25 search with cross-encoder reranking, improving context precision over naive vector search (measured empirically with RAGAS)
- **Dual-backend design** — a custom retrieval pipeline and an AWS Bedrock Knowledge Base sit behind a single interface, switchable with a one-line toggle
- **Stateful LangGraph orchestration** — manages multi-step reasoning and tool calls (filing retrieval, earnings lookups via yFinance) as a graph rather than a single prompt chain
- **Metadata-filtered retrieval** across multiple tickers, so queries stay scoped to the right company and filing

## Managed vs. custom: a real tradeoff analysis

Rather than assuming a managed retrieval service is "good enough" or that a custom pipeline is always better, this project benchmarks the Bedrock Knowledge Base against the custom hybrid pipeline on:

- **Latency**
- **Cost**
- **RAGAS quality scores** (faithfulness, context precision, answer relevancy)

The goal was to ground that decision in measured data instead of a guess — the kind of tradeoff call that matters when choosing infrastructure in a real production system.

## Stack

`Python` · `LangGraph` · `ChromaDB` · `AWS Bedrock Knowledge Base` · `yFinance` · `Streamlit` · `Docker` · `LangSmith` · `RAGAS`

## Running locally

```bash
git clone https://github.com/Daniel-Pasumarthi/financial-research-agent.git
cd financial-research-agent
cp .env.example .env   # add your API keys
docker-compose up --build
```

## Evaluation

Retrieval quality and agent responses are evaluated on-demand using **RAGAS**, with full run tracing via **LangSmith**, so every benchmark result is reproducible and inspectable — not a one-off number.
