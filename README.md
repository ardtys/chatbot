# Basis

Internal knowledge base assistant menggunakan Retrieval-Augmented Generation (RAG) untuk menjawab pertanyaan berdasarkan dokumen perusahaan.

## Features

- **PDF Document Processing** - Upload dan proses dokumen PDF dengan text extraction
- **Semantic Search** - Pencarian berbasis makna menggunakan vector embeddings
- **Contextual Q&A** - Jawaban akurat berdasarkan konteks dokumen
- **Source Transparency** - Menampilkan sumber dokumen yang digunakan untuk menjawab
- **Bilingual Support** - Mendukung pertanyaan dalam Bahasa Indonesia dan Inggris
- **Mobile Responsive** - UI yang optimal di desktop dan mobile

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | React, Vite, TailwindCSS |
| Backend | Python (Vercel Serverless Functions) |
| Vector Database | Upstash Vector |
| Embeddings | Cohere (embed-english-light-v3.0) |
| LLM | Groq Cloud (Llama 3.3 70B) |
| Deployment | Vercel |

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│  Vercel API │────▶│    Groq     │
│   (React)   │     │  (Python)   │     │   (LLM)     │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
                          ▼
                   ┌─────────────┐     ┌─────────────┐
                   │   Cohere    │     │   Upstash   │
                   │ (Embedding) │     │  (Vector)   │
                   └─────────────┘     └─────────────┘
```

**Flow:**
1. User mengajukan pertanyaan
2. Pertanyaan di-embed menggunakan Cohere API
3. Vector search di Upstash untuk menemukan dokumen relevan
4. Konteks dokumen + pertanyaan dikirim ke Groq LLM
5. Jawaban ditampilkan beserta sumber dokumen

## Project Structure

```
rag-system/
├── api/
│   ├── query.py          # Main Q&A endpoint
│   ├── health.py         # Health check endpoint
│   └── requirements.txt  # API dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx       # Main React component
│   │   ├── main.jsx      # Entry point
│   │   └── index.css     # Styles
│   ├── index.html
│   └── package.json
├── scripts/
│   ├── ingest_to_upstash.py  # Document ingestion script
│   └── requirements.txt
├── docs/                 # PDF documents folder
├── vercel.json
└── .env
```

## Setup

### Prerequisites

- Node.js 18+
- Python 3.11+
- Akun [Upstash](https://upstash.com) (Vector Database)
- API Key [Cohere](https://cohere.com) (Embeddings)
- API Key [Groq](https://groq.com) (LLM)

### 1. Clone Repository

```bash
git clone <repository-url>
cd rag-system
```

### 2. Environment Variables

Buat file `.env` di root project:

```env
# Groq Cloud
GROQ_API_KEY=your_groq_api_key
CHAT_MODEL=llama-3.3-70b-versatile

# Cohere
COHERE_API_KEY=your_cohere_api_key

# Upstash Vector
UPSTASH_VECTOR_REST_URL=your_upstash_url
UPSTASH_VECTOR_REST_TOKEN=your_upstash_token

# RAG Configuration
TOP_K=5
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# Paths
DOCS_DIR=./docs
```

### 3. Setup Upstash Vector

1. Buat akun di [Upstash Console](https://console.upstash.com)
2. Buat Vector Index baru dengan konfigurasi:
   - **Dimensions:** 384
   - **Similarity Metric:** Cosine
3. Copy REST URL dan Token ke `.env`

### 4. Install Dependencies

```bash
# Frontend
cd frontend
npm install

# Scripts (untuk ingestion)
cd ../scripts
pip install -r requirements.txt
```

### 5. Ingest Documents

Letakkan file PDF di folder `docs/`, lalu jalankan:

```bash
cd scripts
python ingest_to_upstash.py
```

### 6. Run Development Server

```bash
cd frontend
npm run dev
```

Buka http://localhost:5173

## Deployment (Vercel)

### 1. Push ke GitHub

```bash
git add -A
git commit -m "initial commit"
git push
```

### 2. Deploy di Vercel

1. Import project dari GitHub di [Vercel Dashboard](https://vercel.com/new)
2. Tambahkan Environment Variables:

| Variable | Value |
|----------|-------|
| `GROQ_API_KEY` | Your Groq API key |
| `COHERE_API_KEY` | Your Cohere API key |
| `UPSTASH_VECTOR_REST_URL` | Your Upstash URL |
| `UPSTASH_VECTOR_REST_TOKEN` | Your Upstash token |
| `CHAT_MODEL` | llama-3.3-70b-versatile |
| `TOP_K` | 5 |

3. Deploy

## API Endpoints

### POST /api/query

Menjawab pertanyaan berdasarkan dokumen.

**Request:**
```json
{
  "question": "Apa saja jenis cuti yang tersedia?"
}
```

**Response:**
```json
{
  "answer": "Berdasarkan dokumen, jenis cuti yang tersedia adalah...",
  "sources": [
    {
      "content": "...",
      "source": "kebijakan-cuti.pdf",
      "page": 2,
      "relevance_score": 0.89
    }
  ],
  "question": "Apa saja jenis cuti yang tersedia?"
}
```

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "groq_configured": true
}
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `CHAT_MODEL` | Model LLM yang digunakan | llama-3.3-70b-versatile |
| `TOP_K` | Jumlah dokumen yang di-retrieve | 5 |
| `CHUNK_SIZE` | Ukuran chunk dokumen | 800 |
| `CHUNK_OVERLAP` | Overlap antar chunk | 100 |

## Limitations

- Hanya mendukung PDF text-based (bukan scanned/image PDF)
- Embedding model menggunakan bahasa Inggris (tetap bekerja untuk Bahasa Indonesia)
- Maksimal dokumen tergantung limit Upstash Vector (10,000 vectors untuk free tier)

## License

MIT
