#!/usr/bin/env python3
import os
import sys
import hashlib
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from upstash_vector import Index

load_dotenv()

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_VECTOR_REST_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def get_embeddings():
    print(f"Loading embedding model: {EMBED_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


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


def ingest_to_upstash(chunks, embeddings, index):
    print(f"\nIngesting {len(chunks)} chunks to Upstash Vector...")
    batch_size = 100
    total_ingested = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = []

        for chunk in batch:
            content = chunk.page_content
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)

            embedding = embeddings.embed_query(content)
            doc_id = generate_doc_id(content, source, page)

            vectors.append({
                "id": doc_id,
                "vector": embedding,
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

    print("Connecting to Upstash Vector...")
    index = Index(
        url=UPSTASH_VECTOR_REST_URL,
        token=UPSTASH_VECTOR_REST_TOKEN
    )

    try:
        info = index.info()
        print(f"  -> Connected! Current vectors: {info.vector_count}")
        print(f"  -> Dimension: {info.dimension}")
    except Exception as e:
        print(f"Error connecting to Upstash: {e}")
        return

    chunks = load_and_split_pdfs(DOCS_DIR)
    if not chunks:
        print("No documents to ingest")
        return

    embeddings = get_embeddings()
    ingest_to_upstash(chunks, embeddings, index)

    info = index.info()
    print(f"\nFinal vector count: {info.vector_count}")


if __name__ == "__main__":
    main()
