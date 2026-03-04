from http.server import BaseHTTPRequestHandler
import json
import os
import httpx
import traceback
from groq import Groq


def get_embedding(text):
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        raise ValueError("COHERE_API_KEY not configured")

    response = httpx.post(
        "https://api.cohere.ai/v1/embed",
        headers={
            "Authorization": f"Bearer {cohere_key}",
            "Content-Type": "application/json"
        },
        json={
            "texts": [text],
            "model": "embed-english-light-v3.0",
            "input_type": "search_query",
            "truncate": "END"
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body)

            question = data.get('question', '')

            if not question:
                self._send_error(400, "Question is required")
                return

            if not os.getenv("GROQ_API_KEY"):
                self._send_error(503, "GROQ_API_KEY not configured")
                return

            if not os.getenv("UPSTASH_VECTOR_REST_URL"):
                self._send_error(503, "Upstash Vector not configured")
                return

            if not os.getenv("COHERE_API_KEY"):
                self._send_error(503, "COHERE_API_KEY not configured")
                return

            from upstash_vector import Index

            index = Index(
                url=os.getenv("UPSTASH_VECTOR_REST_URL"),
                token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
            )

            question_embedding = get_embedding(question)

            top_k = int(os.getenv("TOP_K", "5"))
            results = index.query(
                vector=question_embedding,
                top_k=top_k,
                include_metadata=True
            )

            sources = []
            context_parts = []

            if hasattr(results, '__iter__'):
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
                            "relevance_score": score
                        })

            context = "\n\n---\n\n".join(context_parts) if context_parts else "Tidak ada dokumen yang ditemukan."

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

            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            chat_response = client.chat.completions.create(
                model=os.getenv("CHAT_MODEL", "llama-3.3-70b-versatile"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )

            answer = chat_response.choices[0].message.content

            result = {
                "answer": answer,
                "sources": sources,
                "question": question
            }

            self._send_json(200, result)

        except Exception as e:
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            self._send_error(500, error_detail)

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
