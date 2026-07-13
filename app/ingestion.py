import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

CHROMA_DIR = "data/chroma_db"

def ingest_filing(filepath: str, ticker: str) -> None:
    """Load a 10-K, chunk it, embed it, store vectors in ChromaDB."""
    documents = TextLoader(filepath).load()
    
    for doc in documents:
        doc.page_content = BeautifulSoup(doc.page_content, "html.parser").get_text(separator=" ", strip=True)
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["ticker"] = ticker

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"Ingested {len(chunks)} chunks for {ticker}")


if __name__ == "__main__":
    ingest_filing("data/filings/AAPL_10K.txt", "AAPL")