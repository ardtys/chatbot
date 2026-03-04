#!/usr/bin/env python3
import os
import sys
import hashlib
import httpx
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from upstash_vector import Index

load_dotenv()

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_VECTOR_REST_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def get_cohere_embeddings(texts):
    response = httpx.post(
        "https://api.cohere.ai/v1/embed",
        headers={
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "texts": texts,
            "model": "embed-english-light-v3.0",
            "input_type": "search_document",
            "truncate": "END"
        },
        timeout=60
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def load_and_split_pdfs(docs_path):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pdf_files = list(Path(docs_path).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {docs_path}")
        return []

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            for page in pages:
                page.metadata["source"] = pdf_path.name
                page.metadata["page"] = page.metadata.get("page", 0) + 1

            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)} chunks from {pdf_path.name}")
        except Exception as e:
            print(f"  [ERROR] Failed: {e}")

    return all_chunks


def generate_doc_id(content, source, page):
    content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
    return f"{source}-p{page}-{content_hash}"


def ingest_to_upstash(chunks, index):
    print(f"\nIngesting {len(chunks)} chunks to Upstash Vector...")
    batch_size = 96
    total_ingested = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk.page_content for chunk in batch]

        embeddings = get_cohere_embeddings(texts)

        vectors = []
        for j, chunk in enumerate(batch):
            content = chunk.page_content
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)
            doc_id = generate_doc_id(content, source, page)

            vectors.append({
                "id": doc_id,
                "vector": embeddings[j],
                "metadata": {
                    "content": content,
                    "source": source,
                    "page": page
                }
            })

        index.upsert(vectors=vectors)
        total_ingested += len(batch)
        print(f"  -> Ingested {total_ingested}/{len(chunks)} chunks")

    print(f"\n[OK] Successfully ingested {total_ingested} chunks")


def main():
    if not UPSTASH_VECTOR_REST_URL or not UPSTASH_VECTOR_REST_TOKEN:
        print("Error: UPSTASH_VECTOR_REST_URL and UPSTASH_VECTOR_REST_TOKEN required")
        return

    if not COHERE_API_KEY:
        print("Error: COHERE_API_KEY required")
        print("Get free API key at: https://dashboard.cohere.com/api-keys")
        return

    print("Connecting to Upstash Vector...")
    index = Index(
        url=UPSTASH_VECTOR_REST_URL,
        token=UPSTASH_VECTOR_REST_TOKEN
    )

    try:
        info = index.info()
        print(f"  -> Connected! Current vectors: {info.vector_count}")
        print(f"  -> Dimension: {info.dimension}")

        if info.vector_count > 0:
            print(f"\nClearing existing {info.vector_count} vectors...")
            index.reset()
            print("  -> Cleared!")

    except Exception as e:
        print(f"Error connecting to Upstash: {e}")
        return

    chunks = load_and_split_pdfs(DOCS_DIR)
    if not chunks:
        print("No documents to ingest")
        return

    ingest_to_upstash(chunks, index)

    info = index.info()
    print(f"\nFinal vector count: {info.vector_count}")


if __name__ == "__main__":
    main()
