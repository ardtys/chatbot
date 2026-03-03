#!/bin/bash
set -e

echo "=========================================="
echo "  RAG Knowledge Base - Setup & Launch"
echo "=========================================="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker found"

# Check docs
if [ ! -d "docs" ] || [ -z "$(ls docs/*.pdf 2>/dev/null)" ]; then
    echo "⚠️  No PDF files found in docs/ directory"
    echo "   Please add your PDF documents to the docs/ folder"
    exit 1
fi

PDF_COUNT=$(ls docs/*.pdf 2>/dev/null | wc -l)
echo "✅ Found $PDF_COUNT PDF document(s) in docs/"

# Build & Start
echo ""
echo "🚀 Starting all services..."
echo "   This may take 5-10 minutes on first run (downloading models)"
echo ""

docker compose up -d --build

echo ""
echo "⏳ Waiting for services to be ready..."
echo ""

# Wait for backend health
MAX_RETRIES=60
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend ready"
        break
    fi
    RETRY=$((RETRY + 1))
    echo "   Waiting for backend... ($RETRY/$MAX_RETRIES)"
    sleep 5
done

if [ $RETRY -eq $MAX_RETRIES ]; then
    echo "⚠️  Backend took too long. Check logs: docker compose logs backend"
fi

echo ""
echo "=========================================="
echo "  🎉 System Ready!"
echo "=========================================="
echo ""
echo "  Frontend:  http://localhost:3000"
echo "  API:       http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
echo "  Ollama:    http://localhost:11434"
echo ""
echo "  Logs:      docker compose logs -f"
echo "  Stop:      docker compose down"
echo "=========================================="
