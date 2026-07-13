import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()

CHROMA_DIR = "data/chroma_db"

_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings)
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

_bm25_index = None
_bm25_corpus = None

def _get_bm25_index():
    """Build the BM25 index once, then reuse it on every call."""
    global _bm25_index, _bm25_corpus
    if _bm25_index is None:
        stored = _store.get()
        _bm25_corpus = stored["documents"]
        tokenized_corpus = [doc.lower().split() for doc in _bm25_corpus]
        _bm25_index = BM25Okapi(tokenized_corpus)
    return _bm25_index, _bm25_corpus

def _vector_search(query: str, k: int = 10) -> list[str]:
    results = _store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

def _bm25_search(query: str, k: int = 10) -> list[str]:
    bm25_index, corpus = _get_bm25_index()
    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [corpus[i] for i in top_indices]

def retrieve_custom(query: str, top_n: int = 4) -> list[str]:
    """Retrieve wide with hybrid search (vector + BM25), then rerank to top_n."""
    vector_hits = _vector_search(query, k=10)
    bm25_hits = _bm25_search(query, k=10)

    candidates = list(dict.fromkeys(vector_hits + bm25_hits))

    pairs = [(query, candidate) for candidate in candidates]
    scores = _reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_n]]

if __name__ == "__main__":
    results = retrieve_custom("What are the main revenue risks?")
    for i, chunk in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(chunk[:300])
        print()