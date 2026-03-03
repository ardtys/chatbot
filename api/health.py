from http.server import BaseHTTPRequestHandler
import json
import os


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        groq_configured = bool(os.getenv("GROQ_API_KEY"))
        upstash_configured = bool(
            os.getenv("UPSTASH_VECTOR_REST_URL") and
            os.getenv("UPSTASH_VECTOR_REST_TOKEN")
        )

        response = {
            "status": "healthy" if groq_configured else "degraded",
            "groq_configured": groq_configured,
            "upstash_configured": upstash_configured,
            "vectorstore_ready": upstash_configured,
            "documents_ingested": 0,
            "platform": "vercel_serverless"
        }

        if upstash_configured:
            try:
                from upstash_vector import Index
                index = Index(
                    url=os.getenv("UPSTASH_VECTOR_REST_URL"),
                    token=os.getenv("UPSTASH_VECTOR_REST_TOKEN")
                )
                info = index.info()
                response["documents_ingested"] = info.vector_count
                response["vectorstore_ready"] = True
            except Exception as e:
                response["vectorstore_ready"] = False
                response["error"] = str(e)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
