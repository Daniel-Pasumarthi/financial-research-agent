import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

CHROMA_DIR = "data/chroma_db"

_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
_vectorstore = None

def _get_vectorstore():
    """Create the Chroma client once per process, then reuse it for every ingest call."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=_embeddings)
    return _vectorstore

def ingest_filing(filepath: str, ticker: str) -> None:
    """Load a 10-K, chunk it, embed it, store vectors in ChromaDB."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    documents = [Document(page_content=raw_text, metadata={"source": filepath})]

    for doc in documents:
        soup = BeautifulSoup(doc.page_content, "html.parser")
        for header_block in soup.find_all("ix:header"):
            header_block.decompose()
        doc.page_content = soup.get_text(separator=" ", strip=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["ticker"] = ticker

    vectorstore = _get_vectorstore()
    vectorstore.add_documents(chunks)

    print(f"Ingested {len(chunks)} chunks for {ticker}")


if __name__ == "__main__":
    ingest_filing("data/filings/AAPL_10K_primary.html", "AAPL")