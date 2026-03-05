"""
Local Development Server untuk RAG System
Menjalankan backend API menggunakan Upstash Vector + Groq + Cohere
Tanpa memerlukan Docker atau ChromaDB
"""
import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import dependencies
try:
    import httpx
    from upstash_vector import Index
except ImportError:
    print("Installing required packages...")
    os.system("pip install httpx upstash-vector python-dotenv")
    import httpx
    from upstash_vector import Index

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_VECTOR_REST_URL")
UPSTASH_VECTOR_REST_TOKEN = os.getenv("UPSTASH_VECTOR_REST_TOKEN")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile")
TOP_K = int(os.getenv("TOP_K", "5"))

print(f"""
================================================================
         RAG System - Local Development Server
================================================================
  Configuration:
  - Groq API Key: {'[OK] Configured' if GROQ_API_KEY else '[X] Missing'}
  - Cohere API Key: {'[OK] Configured' if COHERE_API_KEY else '[X] Missing'}
  - Upstash URL: {'[OK] Configured' if UPSTASH_VECTOR_REST_URL else '[X] Missing'}
  - Model: {CHAT_MODEL}
  - Top K: {TOP_K}
================================================================
""")


def get_embedding(text):
    """Get embedding from Cohere API"""
    response = httpx.post(
        "https://api.cohere.ai/v1/embed",
        headers={
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "texts": [text],
            "model": "embed-english-light-v3.0",
            "input_type": "search_query"
        },
        timeout=30
    )
    if response.status_code != 200:
        raise Exception(f"Cohere API error: {response.text}")
    return response.json()["embeddings"][0]


def call_groq(prompt):
    """Call Groq LLM API"""
    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000
        },
        timeout=60
    )
    if response.status_code != 200:
        raise Exception(f"Groq API error: {response.text}")
    return response.json()["choices"][0]["message"]["content"]


class RequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self._send_json({})

    def do_GET(self):
        if self.path == '/health' or self.path == '/api/health':
            # Health check
            try:
                index = Index(url=UPSTASH_VECTOR_REST_URL, token=UPSTASH_VECTOR_REST_TOKEN)
                info = index.info()
                vector_count = info.vector_count
            except:
                vector_count = 0

            is_ready = bool(GROQ_API_KEY and COHERE_API_KEY and UPSTASH_VECTOR_REST_URL and vector_count > 0)
            self._send_json({
                "status": "healthy" if is_ready else "degraded",
                "groq_configured": bool(GROQ_API_KEY),
                "cohere_configured": bool(COHERE_API_KEY),
                "upstash_configured": bool(UPSTASH_VECTOR_REST_URL),
                "documents_ingested": vector_count,
                "vectorstore_ready": vector_count > 0,
                "ollama_connected": False,
                "message": "All APIs configured" if (GROQ_API_KEY and COHERE_API_KEY) else "Missing API keys"
            })

        elif self.path == '/documents' or self.path == '/api/documents':
            # List documents
            try:
                index = Index(url=UPSTASH_VECTOR_REST_URL, token=UPSTASH_VECTOR_REST_TOKEN)
                info = index.info()
                self._send_json({
                    "total_vectors": info.vector_count,
                    "dimension": info.dimension
                })
            except Exception as e:
                self._send_json({"error": str(e)}, 500)

        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == '/query' or self.path == '/api/query':
            try:
                content_length = int(self.headers['Content-Length'])
                body = self.rfile.read(content_length)
                data = json.loads(body)

                question = data.get('question', '').strip()
                if not question:
                    self._send_json({"error": "Question is required"}, 400)
                    return

                print(f"\n[QUERY] Question: {question}")

                # 1. Get embedding for question
                print("   -> Getting embedding from Cohere...")
                question_embedding = get_embedding(question)

                # 2. Search Upstash Vector
                print("   -> Searching Upstash Vector...")
                index = Index(url=UPSTASH_VECTOR_REST_URL, token=UPSTASH_VECTOR_REST_TOKEN)
                results = index.query(
                    vector=question_embedding,
                    top_k=TOP_K,
                    include_metadata=True
                )

                # 3. Build context from results
                sources = []
                context_parts = []
                for result in results:
                    metadata = getattr(result, 'metadata', None) or {}
                    content = metadata.get('content', '')
                    source = metadata.get('source', 'Unknown')
                    page = metadata.get('page', 0)
                    score = getattr(result, 'score', 0)

                    if content:
                        context_parts.append(content)
                        sources.append({
                            "content": content,
                            "source": source,
                            "page": page,
                            "relevance_score": round(score, 4)
                        })

                context = "\n\n---\n\n".join(context_parts)
                print(f"   -> Found {len(sources)} relevant chunks")

                # 4. Build prompt
                prompt = f"""Anda adalah asisten knowledge base perusahaan. Jawab pertanyaan berdasarkan dokumen berikut.

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

                # 5. Call Groq LLM
                print("   -> Generating answer with Groq LLM...")
                answer = call_groq(prompt)
                print(f"   -> [OK] Answer generated ({len(answer)} chars)")

                self._send_json({
                    "answer": answer,
                    "sources": sources,
                    "question": question
                })

            except Exception as e:
                print(f"   -> [ERROR] {str(e)}")
                self._send_json({"error": str(e)}, 500)

        else:
            self._send_json({"error": "Not found"}, 404)

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def run_server(port=8000):
    server = HTTPServer(('0.0.0.0', port), RequestHandler)
    print(f"[SERVER] Running at http://localhost:{port}")
    print(f"   - Health: http://localhost:{port}/health")
    print(f"   - Query:  POST http://localhost:{port}/query")
    print(f"\n[INFO] Press Ctrl+C to stop the server\n")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n[INFO] Server stopped")
        server.shutdown()


if __name__ == "__main__":
    run_server(8000)
