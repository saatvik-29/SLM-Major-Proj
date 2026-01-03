# Generic Self-Hosted Support SLM

A fully offline, self-hosted support assistant that answers questions strictly from uploaded documents using a local Small Language Model (SLM) and RAG.

## Features
- **Offline & Private**: Self-hosted, no data leaves your machine.
- **Generic SLM**: Uses [Phi-3 Mini 4k Instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) (downloaded automatically).
- **Document Support**: PDF, DOCX, TXT, MD.
- **API First**: FastAPI backend with Swagger docs at `/docs`.
- **Easy Deployment**: Dockerized with `docker-compose`.

## Setup

### Prerequisites
- Docker & Docker Compose
- OR Python 3.9+ (for local run)

### Running with Docker (Recommended)
```bash
docker compose up --build
```
Access the UI at `http://localhost:8000/ui/web/index.html` (Note: In this simple setup, you might need to open the file directly or serve it. The current setup serves the API at 8000. For simplicity, open `ui/web/index.html` in your browser directly, it is configured to hit `localhost:8000`).

*Wait for the model to download on the first run (approx 2.4GB).*

### Running Locally
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python -m backend.main
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Upload Documents
```bash
curl -X POST "http://localhost:8000/v1/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/path/to/manual.pdf"
```

### Ask a Question
```bash
curl -X POST "http://localhost:8000/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reset the device?"}'
```

### Reset Index
```bash
curl -X POST "http://localhost:8000/v1/reset"
```

## Architecture
- **Ingestion**: Text extraction -> Chunking (400 tokens) -> Embeddings (all-MiniLM-L6-v2) -> FAISS Index.
- **RAG**: Top-3 retrieval -> Context injection.
- **Inference**: Llama-cpp-python running Phi-3 GGUF.

## License
MIT
