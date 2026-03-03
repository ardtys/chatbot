from http.server import BaseHTTPRequestHandler
import json
import os


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            upstash_url = os.getenv("UPSTASH_VECTOR_REST_URL")
            upstash_token = os.getenv("UPSTASH_VECTOR_REST_TOKEN")

            if not upstash_url or not upstash_token:
                self._send_json(200, {
                    "documents": [],
                    "total": 0,
                    "vectorstore_ready": False
                })
                return

            from upstash_vector import Index

            index = Index(url=upstash_url, token=upstash_token)
            info = index.info()

            response = {
                "documents": [],
                "total": info.vector_count,
                "vectorstore_ready": True,
                "index_info": {
                    "vector_count": info.vector_count,
                    "dimension": info.dimension,
                    "similarity_function": info.similarity_function
                }
            }

            self._send_json(200, response)

        except Exception as e:
            self._send_json(500, {
                "documents": [],
                "total": 0,
                "vectorstore_ready": False,
                "error": str(e)
            })

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _send_json(self, status, data):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
