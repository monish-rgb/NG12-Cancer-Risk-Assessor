"""RAG pipeline — query ChromaDB for relevant NG12 guideline chunks."""

import os
import sys

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
COLLECTION_NAME = "ng12_guidelines"
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")

_collection = None
_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
        )
    return _embeddings


def _auto_ingest_pdfs():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ingestion.ingest_pdf import extract_pages, chunk_text, create_embeddings, build_vectorstore

    persist_dir = os.path.abspath(VECTORSTORE_PATH)
    data_dir = os.path.abspath(DATA_FOLDER)

    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in data folder.")
        return

    all_chunks = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        print(f"[Ingest] Processing: {pdf_file}")
        try:
            pages = extract_pages(pdf_path)
            chunks = chunk_text(pages)
            print(f"  {len(pages)} pages, {len(chunks)} chunks")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  Error processing {pdf_file}: {e}")

    if all_chunks:
        print(f"[Ingest] Embedding {len(all_chunks)} chunks...")
        embeddings = create_embeddings(all_chunks)
        os.makedirs(persist_dir, exist_ok=True)
        build_vectorstore(all_chunks, embeddings, persist_dir)
        print("[Ingest] Vector store ready.")


def _get_collection():
    global _collection
    if _collection is not None:
        return _collection

    persist_dir = os.path.abspath(VECTORSTORE_PATH)
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            _collection = collection
            return _collection
        raise ValueError("empty")
    except (ValueError, Exception):
        print("[RAG] Vector store empty or missing. Starting ingestion...")
        _auto_ingest_pdfs()
        client = chromadb.PersistentClient(path=persist_dir)
        _collection = client.get_collection(COLLECTION_NAME)
        print(f"[RAG] Loaded {_collection.count()} chunks.")
        return _collection


def embed_query(query: str) -> list[float]:
    return _get_embeddings().embed_query(query)


def _query(query_embedding: list[float], top_k: int) -> list[dict]:
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "chunk_id": results["ids"][0][i],
            "page": results["metadatas"][0][i]["page"],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]


def query_guidelines(symptoms: list[str], top_k: int = 5) -> list[dict]:
    query_text = "Cancer referral guidelines for symptoms: " + ", ".join(symptoms)
    return _query(embed_query(query_text), top_k)


def query_guidelines_text(query: str, top_k: int = 5) -> list[dict]:
    return _query(embed_query(query), top_k)
