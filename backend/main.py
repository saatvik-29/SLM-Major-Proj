from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import shutil
import os

from .ingest import DocumentIngester
from .rag import RAGEngine
from .inference import SLMEngine
from .utils import get_logger, timer

logger = get_logger("api")

app = FastAPI(title="Generic Self-Hosted Support SLM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_engine = RAGEngine()
ingester = DocumentIngester()
# Lazy load SLM to avoid startup delay if just checking API, 
# but requirement says "Model downloaded once on startup". 
# So we initialize it on startup.
slm_engine = None

@app.on_event("startup")
async def startup_event():
    global slm_engine
    # Initialize SLM (will download if needed)
    slm_engine = SLMEngine()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.get("/health")
def health_check():
    return {"status": "ok", "components": {"rag": "ready", "slm": "ready" if slm_engine else "loading"}}

@app.post("/v1/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    total_chunks = 0
    processed_files = []
    
    with timer("Document Upload & Indexing"):
        for file in files:
            logger.info(f"Processing {file.filename}...")
            text = await ingester.extract_text(file)
            if text:
                chunks = ingester.chunk_text(text, source=file.filename)
                rag_engine.add_documents(chunks)
                total_chunks += len(chunks)
                processed_files.append(file.filename)
            else:
                logger.warning(f"No text extracted from {file.filename}")

    return {
        "status": "success",
        "files_processed": processed_files,
        "chunks_indexed": total_chunks
    }

@app.post("/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    with timer(f"Query: {request.question}"):
        # Retrieve (k=2 for faster, leaner context)
        retrieved_docs = rag_engine.search(request.question, k=2)
        
        if not retrieved_docs:
            return {
                "answer": "I don't know based on the provided documents.",
                "sources": []
            }
        
        # Construct Context — trim each chunk to keep prompt small
        MAX_CHUNK_CHARS = 500
        context_text = "\n\n".join(
            [f"Source: {d['source']}\nContent: {d['text'][:MAX_CHUNK_CHARS]}" for d in retrieved_docs]
        )
        sources = list(set([d['source'] for d in retrieved_docs]))
        
        # Generate
        answer = slm_engine.generate_answer(context_text, request.question)
        
        return {
            "answer": answer,
            "sources": sources
        }

@app.post("/v1/reset")
def reset_index():
    rag_engine.reset()
    return {"status": "success", "message": "Index reset"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
