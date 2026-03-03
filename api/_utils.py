import os
import hashlib


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_VECTOR_REST_URL", "")
UPSTASH_VECTOR_REST_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN", "")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile")
TOP_K = int(os.getenv("TOP_K", "5"))


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def get_llm():
    from langchain_groq import ChatGroq
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is required")
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=CHAT_MODEL,
        temperature=0.1,
        max_tokens=1000,
    )


def get_vector_store():
    from upstash_vector import Index
    if not UPSTASH_VECTOR_REST_URL or not UPSTASH_VECTOR_REST_TOKEN:
        return None
    return Index(
        url=UPSTASH_VECTOR_REST_URL,
        token=UPSTASH_VECTOR_REST_TOKEN
    )


def generate_doc_id(content, source, page):
    content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
    return f"{source}-p{page}-{content_hash}"


SYSTEM_PROMPT = """Anda adalah asisten knowledge base perusahaan. Jawab pertanyaan berdasarkan dokumen berikut.

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

JAWABAN:"""
