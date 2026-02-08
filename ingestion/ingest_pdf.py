"""
PDF Ingestion Script for NG12 Cancer Guidelines.

Parses the NG12 PDF, chunks text with page metadata,
generates embeddings via Google Gemini, and stores in ChromaDB.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

import chromadb
import fitz  # PyMuPDF
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "suspected-cancer-recognition-and-referral-pdf-1837268071621.pdf",
)
VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
COLLECTION_NAME = "ng12_guidelines"
CHUNK_SIZE = 500  # approximate tokens (chars / 4)
CHUNK_OVERLAP = 100


def extract_pages(pdf_path: str) -> list[dict]:
    """Extract text from each page of the PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def chunk_text(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split page text into overlapping chunks, preserving page metadata."""
    chunks = []
    chunk_counter = 0

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]
        # Use character-based chunking (approx 4 chars per token)
        char_chunk_size = chunk_size * 4
        char_overlap = overlap * 4

        start = 0
        while start < len(text):
            end = start + char_chunk_size
            chunk_text_content = text[start:end]

            if chunk_text_content.strip():
                chunk_id = f"ng12_p{page_num:03d}_c{chunk_counter:04d}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_num,
                    "text": chunk_text_content.strip(),
                })
                chunk_counter += 1

            start += char_chunk_size - char_overlap

    return chunks


def create_embeddings(chunks: list[dict]) -> list[list[float]]:
    """Generate embeddings for chunks using gemini embedding model.
    """
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    embeddings = []
    batch_size = 20

    for i in range(0, len(chunks), batch_size):
        batch_texts = [c["text"] for c in chunks[i : i + batch_size]]
        batch_num = i // batch_size + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        print(f"  Embedding batch {batch_num}/{total_batches}...")

        batch_embeddings = embeddings_model.embed_documents(batch_texts)
        embeddings.extend(batch_embeddings)

    return embeddings


def build_vectorstore(chunks: list[dict], embeddings: list[list[float]], persist_dir: str):
    """Store chunks and embeddings in ChromaDB."""
    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"page": c["page"], "chunk_id": c["chunk_id"]} for c in chunks],
    )

    print(f"  Stored {len(chunks)} chunks in ChromaDB at {persist_dir}")
    return collection


def main():
    parser = argparse.ArgumentParser(description="Ingest NG12 PDF into ChromaDB vector store")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion even if collection exists")
    parser.add_argument("--pdf", default=PDF_PATH, help="Path to NG12 PDF")
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf)
    persist_dir = os.path.abspath(VECTORSTORE_PATH)

    # Check if already ingested
    if not args.force and os.path.exists(persist_dir):
        try:
            client = chromadb.PersistentClient(path=persist_dir)
            collection = client.get_collection(COLLECTION_NAME)
            count = collection.count()
            if count > 0:
                print(f"Vector store already exists with {count} chunks. Use --force to re-ingest.")
                return
        except (ValueError, Exception):
            pass

    # Validate PDF exists
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found at {pdf_path}")
        print("Please place the NG12 guidelines PDF at: data/suspected-cancer-recognition-and-referral-pdf-1837268071621.pdf")
        print("Download URL: https://www.nice.org.uk/guidance/ng12/resources/suspected-cancer-recognition-and-referral-pdf-1837268071621")
        sys.exit(1)

    # Validate Google API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        sys.exit(1)

    print(f"[1/4] Extracting text from {pdf_path}")
    pages = extract_pages(pdf_path)
    print(f"  Extracted {len(pages)} pages")

    print("[2/4] Chunking text")
    chunks = chunk_text(pages)
    print(f"  Created {len(chunks)} chunks")

    print("[3/4] Generating embeddings via Google Gemini")
    embeddings = create_embeddings(chunks)

    print("[4/4] Building vector store")
    os.makedirs(persist_dir, exist_ok=True)
    build_vectorstore(chunks, embeddings, persist_dir)

    print("Vector store is ready!!")


if __name__ == "__main__":
    main()
