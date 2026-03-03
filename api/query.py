from http.server import BaseHTTPRequestHandler
import json
import os


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)

            question = data.get('question', '')
            chat_history = data.get('chat_history', [])

            if not question:
                self._send_error(400, "Question is required")
                return

            if not os.getenv("GROQ_API_KEY"):
                self._send_error(503, "GROQ_API_KEY not configured")
                return

            if not os.getenv("UPSTASH_VECTOR_REST_URL"):
                self._send_error(503, "Upstash Vector not configured")
                return

            from upstash_vector import Index
            from langchain_groq import ChatGroq
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            index = Index(
                url=os.getenv("UPSTASH_VECTOR_REST_URL"),
                token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
            )

            llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model_name=os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile"),
                temperature=0.1,
                max_tokens=1000,
            )

            question_embedding = embeddings.embed_query(question)

            top_k = int(os.getenv("TOP_K", "5"))
            results = index.query(
                vector=question_embedding,
                top_k=top_k,
                include_metadata=True
            )

            sources = []
            context_parts = []

            for result in results:
                if result.metadata:
                    content = result.metadata.get('content', '')
                    source = result.metadata.get('source', 'Unknown')
                    page = result.metadata.get('page', 0)

                    context_parts.append(content)
                    sources.append({
                        "content": content,
                        "source": source,
                        "page": page,
                        "relevance_score": result.score
                    })

            context = "\n\n---\n\n".join(context_parts)

            prompt = f"""Anda adalah asisten knowledge base perusahaan. Jawab pertanyaan berdasarkan dokumen berikut.

ATURAN FORMAT JAWABAN:
1. Gunakan format yang rapi dan mudah dibaca
2. Gunakan bullet points (•) untuk daftar item
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

            response = llm.invoke(prompt)
            answer = response.content

            result = {
                "answer": answer,
                "sources": sources,
                "question": question
            }

            self._send_json(200, result)

        except Exception as e:
            self._send_error(500, f"Query failed: {str(e)}")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _send_json(self, status, data):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_error(self, status, message):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"detail": message}).encode())
