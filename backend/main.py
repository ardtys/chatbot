import os
import hashlib
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if GROQ_API_KEY:
    from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "mistral")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "4"))

app = FastAPI(title="Basis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore: Optional[Chroma] = None
qa_chain = None
chat_history: list = []
ingested_docs: list = []


class QueryRequest(BaseModel):
    question: str
    chat_history: list[list[str]] = []


class SourceChunk(BaseModel):
    content: str
    source: str
    page: int
    relevance_score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    question: str


class IngestResponse(BaseModel):
    message: str
    documents_processed: int
    total_chunks: int
    documents: list[str]


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    groq_configured: bool
    documents_ingested: int
    vectorstore_ready: bool


def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def get_llm():
    if GROQ_API_KEY:
        logger.info(f"Using Groq Cloud with model: {CHAT_MODEL}")
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=CHAT_MODEL,
            temperature=0.1,
            max_tokens=500,
        )
    else:
        logger.info(f"Using Ollama with model: {CHAT_MODEL}")
        return ChatOllama(
            model=CHAT_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            num_ctx=2048,
            num_predict=150,
            repeat_penalty=1.2,
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
        logger.warning(f"No PDF files found in {docs_path}")
        return []

    for pdf_path in pdf_files:
        logger.info(f"Processing: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            for page in pages:
                page.metadata["source"] = pdf_path.name
                page.metadata["page"] = page.metadata.get("page", 0) + 1

            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            logger.info(f"  -> {len(chunks)} chunks from {pdf_path.name}")
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    return all_chunks


def build_vectorstore(chunks):
    embeddings = get_embeddings()

    ids = []
    for chunk in chunks:
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:12]
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        ids.append(f"{source}-p{page}-{content_hash}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="company_knowledge",
        ids=ids,
    )

    logger.info(f"Vectorstore built with {len(chunks)} chunks")
    return vectorstore


def build_qa_chain(vs):
    llm = get_llm()

    system_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Anda adalah asisten knowledge base perusahaan. Jawab pertanyaan berdasarkan dokumen berikut.

ATURAN FORMAT JAWABAN:
1. Gunakan format yang rapi dan mudah dibaca
2. Gunakan bullet points untuk daftar item
3. Gunakan penomoran (1, 2, 3) untuk langkah-langkah atau prosedur
4. Pisahkan kategori/bagian dengan baris kosong
5. Tebalkan informasi penting dengan **teks**
6. Jangan gunakan markdown header (#)
7. Jawab dalam Bahasa Indonesia yang baik dan benar

ATURAN KONTEN:
- Hanya gunakan informasi dari dokumen yang diberikan
- Jika informasi tidak ada di dokumen, katakan "Informasi tidak ditemukan dalam dokumen"
- Berikan jawaban yang lengkap namun ringkas

DOKUMEN:
{context}

PERTANYAAN: {question}

JAWABAN:""",
    )

    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": system_prompt},
        verbose=False,
    )

    return chain


@app.on_event("startup")
async def startup_ingest():
    global vectorstore, qa_chain, ingested_docs

    if not Path(DOCS_DIR).exists():
        logger.info("No docs directory found, skipping auto-ingest")
        return

    pdf_files = list(Path(DOCS_DIR).glob("*.pdf"))
    if not pdf_files:
        logger.info("No PDFs found in docs/, skipping auto-ingest")
        return

    try:
        logger.info(f"Auto-ingesting {len(pdf_files)} PDFs from {DOCS_DIR}")
        chunks = load_and_split_pdfs(DOCS_DIR)

        if chunks:
            vectorstore = build_vectorstore(chunks)
            qa_chain = build_qa_chain(vectorstore)
            ingested_docs = [f.name for f in pdf_files]
            logger.info("RAG pipeline ready!")
    except Exception as e:
        logger.error(f"Startup ingest failed: {e}")
        logger.info("System will start without vectorstore. Use /ingest to load docs manually.")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    ollama_ok = False
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            ollama_ok = resp.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if vectorstore and (ollama_ok or GROQ_API_KEY) else "degraded",
        ollama_connected=ollama_ok,
        groq_configured=bool(GROQ_API_KEY),
        documents_ingested=len(ingested_docs),
        vectorstore_ready=vectorstore is not None,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    global vectorstore, qa_chain, ingested_docs

    chunks = load_and_split_pdfs(DOCS_DIR)
    if not chunks:
        raise HTTPException(status_code=404, detail="No PDF documents found in docs/ directory")

    vectorstore = build_vectorstore(chunks)
    qa_chain = build_qa_chain(vectorstore)
    ingested_docs = [f.name for f in Path(DOCS_DIR).glob("*.pdf")]

    return IngestResponse(
        message="Documents ingested successfully",
        documents_processed=len(ingested_docs),
        total_chunks=len(chunks),
        documents=ingested_docs,
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vectorstore, qa_chain, ingested_docs

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    save_path = Path(DOCS_DIR) / file.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    save_path.write_bytes(content)
    logger.info(f"Uploaded: {file.filename}")

    chunks = load_and_split_pdfs(DOCS_DIR)
    vectorstore = build_vectorstore(chunks)
    qa_chain = build_qa_chain(vectorstore)
    ingested_docs = [f.name for f in Path(DOCS_DIR).glob("*.pdf")]

    return {
        "message": f"Uploaded {file.filename} and re-indexed",
        "total_documents": len(ingested_docs),
        "total_chunks": len(chunks),
    }


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(req: QueryRequest):
    if not qa_chain:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not initialized. Please ingest documents first via /ingest",
        )

    try:
        formatted_history = [tuple(pair) for pair in req.chat_history if len(pair) == 2]

        result = qa_chain.invoke({
            "question": req.question,
            "chat_history": formatted_history,
        })

        sources = []
        seen = set()
        for doc in result.get("source_documents", []):
            content = doc.page_content.strip()
            content_key = content[:100]
            if content_key in seen:
                continue
            seen.add(content_key)

            sources.append(SourceChunk(
                content=content,
                source=doc.metadata.get("source", "Unknown"),
                page=doc.metadata.get("page", 0),
            ))

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            question=req.question,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/documents")
async def list_documents():
    return {
        "documents": ingested_docs,
        "total": len(ingested_docs),
        "vectorstore_ready": vectorstore is not None,
    }


@app.post("/reset")
async def reset_knowledge_base():
    global vectorstore, qa_chain, ingested_docs, chat_history

    import shutil
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)

    vectorstore = None
    qa_chain = None
    ingested_docs = []
    chat_history = []

    return {"message": "Knowledge base reset successfully"}
