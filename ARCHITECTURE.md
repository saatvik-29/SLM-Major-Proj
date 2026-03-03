# SLM Support Assistant — Architecture & Design Document

> **Last Updated:** March 2026
> **Project:** Self-Hosted, Offline-Capable Document Q&A System
> **Stack:** FastAPI · llama-cpp-python · FAISS · sentence-transformers · Docker (NVIDIA GPU)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [RAG Pipeline Architecture](#2-rag-pipeline-architecture)
3. [Hierarchical Tree Chunking Strategy](#3-hierarchical-tree-chunking-strategy)
4. [Embedding Model](#4-embedding-model)
5. [Vector Store (FAISS)](#5-vector-store-faiss)
6. [Retrieval Strategy](#6-retrieval-strategy)
7. [Small Language Model (SLM)](#7-small-language-model-slm)
8. [Inference Parameters](#8-inference-parameters)
9. [API Design](#9-api-design)
10. [Docker & GPU Configuration](#10-docker--gpu-configuration)
11. [Data Flow — End to End](#11-data-flow--end-to-end)
12. [File Structure](#12-file-structure)
13. [Design Decisions & Trade-offs](#13-design-decisions--trade-offs)
14. [External API Integration](#14-external-api-integration)

---

## 1. System Overview

This project is a **fully self-hosted, offline-capable document Q&A assistant** built on a Retrieval-Augmented Generation (RAG) architecture. Users upload documents (PDF, DOCX, TXT, Markdown) through a web UI. The system indexes their content into a local vector store. When a user asks a question, the system retrieves the most relevant sections of the documents and uses a locally-running Small Language Model (SLM) to synthesize a grounded, factual answer.

### Core Design Principles
- **No external API calls** — all inference runs locally inside Docker
- **Strict grounding** — the SLM is instructed to answer only from retrieved document context
- **Low latency** — model and retrieval parameters are tuned for fast response
- **GPU-accelerated** — the SLM runs on the NVIDIA GPU via CUDA offloading

---

## 2. RAG Pipeline Architecture

The system follows a standard two-phase RAG pipeline with an important enhancement: **Hierarchical Tree Chunking** during the ingestion phase.

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PHASE                          │
│                                                                 │
│  Document Upload                                                │
│       │                                                         │
│       ▼                                                         │
│  Text Extraction (PDF / DOCX / TXT / MD)                        │
│       │                                                         │
│       ▼                                                         │
│  Text Cleaning (whitespace normalization)                       │
│       │                                                         │
│       ▼                                                         │
│  Hierarchical Chunking ──────────────────────────────────────┐  │
│    ├─ Large Parent Chunks (800 words, 150 overlap)           │  │
│    └─ Small Child Chunks (200 words, 50 overlap)             │  │
│         ↑ embedded into FAISS                                │  │
│         ↑ stores reference to parent_text                    │  │
│                                                              │  │
│  Embedding (all-MiniLM-L6-v2, 384-dim) ←────────────────────┘  │
│       │                                                         │
│       ▼                                                         │
│  FAISS IndexFlatL2 (persistent on disk)                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         QUERY PHASE                             │
│                                                                 │
│  User Question                                                  │
│       │                                                         │
│       ▼                                                         │
│  Embed Query (all-MiniLM-L6-v2)                                 │
│       │                                                         │
│       ▼                                                         │
│  FAISS Search (k=6 child chunks searched)                       │
│       │                                                         │
│       ▼                                                         │
│  Hierarchical Retrieval                                         │
│    ├─ Deduplicate by parent_id                                  │
│    └─ Return top k=2 unique parent_texts                        │
│       │                                                         │
│       ▼                                                         │
│  Prompt Construction (system + context + question)              │
│       │                                                         │
│       ▼                                                         │
│  SLM Inference (Llama-3.2-1B-Instruct, GPU)                     │
│       │                                                         │
│       ▼                                                         │
│  Answer + Sources returned to user                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Hierarchical Tree Chunking Strategy

### Why not traditional flat chunking?

Traditional RAG splits documents into fixed-size chunks (e.g., 400 words) and embeds each one directly. This creates a fundamental tension:

| Chunk Size | Embedding Accuracy | Context for SLM |
|---|---|---|
| **Small** | ✅ High (focused, specific) | ❌ Low (too little context) |
| **Large** | ❌ Low (diluted, noisy) | ✅ High (rich context) |

Flat chunking forces you to pick one or the other. For large documents (technical manuals, research papers), this leads to poor retrieval accuracy or context-poor answers.

### The Hierarchical Tree Solution (Parent-Child)

We use a **two-level hierarchical structure**:

```
Document
    │
    ├── Parent Chunk 1 (800 words) ─── used for SLM generation context
    │       ├── Child Chunk 1.1 (200 words) ─── embedded in FAISS
    │       ├── Child Chunk 1.2 (200 words) ─── embedded in FAISS
    │       ├── Child Chunk 1.3 (200 words) ─── embedded in FAISS
    │       └── Child Chunk 1.4 (200 words) ─── embedded in FAISS
    │
    ├── Parent Chunk 2 (800 words)
    │       ├── Child Chunk 2.1 (200 words)
    │       └── ...
    └── ...
```

**Child chunks** are small and precise — they are what gets embedded into FAISS and matched against the user's query. This gives **high retrieval precision**.

**Parent chunks** are large and context-rich — when a child chunk is matched, its parent is retrieved and passed to the SLM. This gives the model **full contextual understanding**.

### Chunking Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `parent_chunk_size` | **800 words** | Large enough to capture a complete topic section |
| `parent_chunk_overlap` | **150 words** | Prevents losing context at chunk boundaries |
| `child_chunk_size` | **200 words** | Small enough for precise semantic matching |
| `child_chunk_overlap` | **50 words** | Avoids edge-of-chunk misses |

### Chunk Metadata Structure

Each child chunk stored in FAISS carries the following fields:

```python
{
    "text": "...",          # Small child text — used for embedding & search
    "parent_text": "...",   # Full parent text — passed to SLM as context
    "source": "filename",   # Original document filename
    "parent_id": "parent_filename_0"  # Used for deduplication
}
```

---

## 4. Embedding Model

| Property | Value |
|---|---|
| **Model** | `all-MiniLM-L6-v2` |
| **Library** | `sentence-transformers` |
| **Embedding Dimension** | 384 |
| **Model Size** | ~90 MB |
| **Inference Device** | CPU (fast enough for encoding; GPU not needed here) |

`all-MiniLM-L6-v2` is chosen for its excellent balance of speed and semantic accuracy. It is a distilled version of MiniLM, trained specifically for semantic sentence similarity, making it ideal for RAG retrieval tasks.

---

## 5. Vector Store (FAISS)

| Property | Value |
|---|---|
| **Library** | `faiss-cpu` |
| **Index Type** | `IndexFlatL2` (exact L2 nearest neighbour) |
| **Dimensions** | 384 (matches embedding model) |
| **Persistence** | Yes — `data/vector_store/index.faiss` + `index.pkl` |
| **Metadata Store** | Python `pickle` alongside FAISS index |

`IndexFlatL2` performs an exact brute-force search. For the expected document sizes in this project (hundreds to low-thousands of chunks), this is faster than approximate methods (like HNSW or IVF) because it avoids the overhead of index building and has zero approximation error.

The index and metadata are persisted to disk on every update, so uploaded documents survive container restarts.

---

## 6. Retrieval Strategy

The retrieval logic uses a **hierarchical deduplication** approach:

1. Query FAISS for `search_k = max(k * 3, 10)` child chunks (deliberately over-fetches)
2. Iterate through results in distance order
3. For each result, check its `parent_id`
4. If the parent has **not** been seen before → add its `parent_text` to results
5. Stop once `k=2` unique parents have been collected

```python
search_k = max(k * 3, 10)   # Fetch 10 candidates to find 2 unique parents
```

This ensures that even if multiple child chunks from the same parent section are the closest matches, the SLM receives **2 distinct, non-redundant context blocks** — preventing repetition and maximising information density per token.

### Context Truncation in API

In `main.py`, each retrieved parent context is **further hard-truncated** to 500 characters when constructing the prompt:

```python
MAX_CHUNK_CHARS = 500
context_text = "\n\n".join(
    [f"Source: {d['source']}\nContent: {d['text'][:MAX_CHUNK_CHARS]}" for d in retrieved_docs]
)
```

> **Note:** This 500-char limit caps the context fed to the SLM, which keeps the prompt compact for lower latency. It can be increased if more context fidelity is needed.

---

## 7. Small Language Model (SLM)

| Property | Value |
|---|---|
| **Model Name** | `Llama-3.2-1B-Instruct` |
| **Quantization** | `Q4_K_M` (4-bit, K-Quant, Medium) |
| **Format** | GGUF |
| **Model Size** | ~800 MB on disk |
| **Source** | [bartowski/Llama-3.2-1B-Instruct-GGUF on HuggingFace](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) |
| **Inference Engine** | `llama-cpp-python` (Python bindings for `llama.cpp`) |
| **Download** | Automatic on first startup if not present |

### Why Llama 3.2 1B?

- **Speed** — At 1B parameters with Q4 quantization, it is extremely fast, especially on GPU
- **Instruction-following** — The `-Instruct` variant is fine-tuned for following system prompts, making it reliable for strict RAG (answer only from context)
- **Recency** — Meta's Llama 3.2 series is a very modern architecture with strong per-parameter efficiency
- **Size** — At ~800 MB, it is small enough to fit entirely in a modest GPU's VRAM (4 GB+)

### Why Q4_K_M Quantization?

`Q4_K_M` is the recommended sweet spot in the GGUF quantization spectrum:

| Quantization | Size | Quality Loss | Speed |
|---|---|---|---|
| Q2_K | Smallest | High | Fastest |
| **Q4_K_M** | **Medium** | **Low** | **Fast** |
| Q8_0 | Large | Minimal | Slower |
| F16 | Largest | None | Slowest |

`Q4_K_M` uses K-Quant which applies non-uniform quantization, preserving the most important weights at higher precision. This makes it significantly better than naive Q4_0 at a negligible size increase.

---

## 8. Inference Parameters

The model is initialized in `backend/inference.py` with the following parameters:

```python
self.llm = Llama(
    model_path=self.model_path,
    n_ctx=2048,           # Context window size
    n_threads=os.cpu_count(), # Use all available CPU threads
    n_batch=512,          # Token batch size for prompt processing
    n_gpu_layers=-1,      # Offload ALL layers to GPU (-1 = all)
    flash_attn=True,      # Enable Flash Attention for faster computation
    verbose=False
)
```

| Parameter | Value | Purpose |
|---|---|---|
| `n_ctx` | `2048` | Maximum tokens in context window. Sufficient for 2 parent chunks + question + answer. |
| `n_threads` | `os.cpu_count()` | Use all CPU threads for non-GPU operations (tokenization, sampling) |
| `n_batch` | `512` | Number of tokens processed in parallel during prompt evaluation |
| `n_gpu_layers` | `-1` | **Offload ALL transformer layers to GPU VRAM** — critical for low latency |
| `flash_attn` | `True` | Flash Attention v2 — reduces VRAM usage and speeds up attention computation |
| `verbose` | `False` | Suppress llama.cpp internal logging |

### Generation Parameters

```python
output = self.llm.create_chat_completion(
    messages=[system_message, user_message],
    max_tokens=256,    # Cap output length
    temperature=0.1,   # Near-greedy decoding for factual answers
)
```

| Parameter | Value | Purpose |
|---|---|---|
| `max_tokens` | `256` | Caps the response length. Keeps latency low and forces concise answers. |
| `temperature` | `0.1` | Near-deterministic/greedy decoding. Ideal for factual Q&A where creativity is undesirable. |

### System Prompt

The SLM is given the following strict system prompt at inference time:

```
You are a helpful support assistant. Answer the user's question 
using ONLY the provided context. Be concise and direct. 
If the answer is not in the context, say 
"I don't know based on the provided documents."
```

This grounds the model completely to the retrieved document context and prevents hallucination.

---

## 9. API Design

The backend is a **FastAPI** application served by **Uvicorn** on port `8000`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns RAG and SLM status |
| `POST` | `/v1/documents/upload` | Upload one or more documents for ingestion |
| `POST` | `/v1/query` | Ask a question, returns answer + source filenames |
| `POST` | `/v1/reset` | Wipe the FAISS index and start fresh |

### Query Request / Response

**Request:**
```json
{
  "question": "What is the return policy?"
}
```

**Response:**
```json
{
  "answer": "The return policy allows returns within 30 days of purchase...",
  "sources": ["policy_manual.pdf"]
}
```

### Startup Behaviour

On container startup, `SLMEngine` is initialized via `@app.on_event("startup")`. This:
1. Checks if the model file exists at `models/slm/Llama-3.2-1B-Instruct-Q4_K_M.gguf`
2. If not, downloads it from HuggingFace automatically
3. Loads it into memory (and GPU VRAM) via `llama.cpp`

---

## 10. Docker & GPU Configuration

### Dockerfile

The container is built on **`nvidia/cuda:12.1.1-runtime-ubuntu22.04`**. This base image:
- Includes the CUDA runtime libraries (`libcudart`, `libcublas`) needed to *run* CUDA programs
- Does NOT include the full developer toolkit (no nvcc, no headers) — keeping the image lean

`llama-cpp-python` is installed from **pre-built CUDA 12.1 wheels** hosted by the maintainer, avoiding source compilation entirely:

```dockerfile
RUN pip install --no-cache-dir \
    llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

> This is important — building from source requires `libcuda.so.1` at compile time, which is only available on the host's NVIDIA driver, not inside the Docker build context.

### docker-compose.yml — GPU Passthrough

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

This passes the host Windows NVIDIA GPU into the Linux container using the **NVIDIA Container Toolkit** (WSL2 GPU passthrough). The `libcuda.so.1` driver library from the host is automatically mounted into the container at runtime, which is why `n_gpu_layers=-1` works correctly at runtime even though it wasn't available at build time.

### Volume Mounts

```yaml
volumes:
  - ./data:/app/data      # FAISS index (persists between restarts)
  - ./models:/app/models  # Downloaded SLM GGUF file (cached)
```

Both critical directories are volume-mounted so data and the model download are **preserved across container rebuilds**.

---

## 11. Data Flow — End to End

### Ingestion Flow

```
User uploads: report.pdf
    │
    ▼
pypdf extracts text from all pages
    │
    ▼
clean_text() normalizes whitespace
    │
    ▼
chunk_text() creates hierarchy:
  - Parent 0 (words 0–799) → id: "parent_report.pdf_0"
      - Child 0.1 (words 0–199)   → embedded as vector #0
      - Child 0.2 (words 150–349) → embedded as vector #1
      - Child 0.3 (words 300–499) → embedded as vector #2
      ....
  - Parent 1 (words 650–1449) → id: "parent_report.pdf_1"
      ....
    │
    ▼
SentenceTransformer encodes all child chunk texts → 384-dim vectors
    │
    ▼
Vectors added to FAISS IndexFlatL2
Metadata (including parent_text) saved to pickle
    │
    ▼
index.faiss + index.pkl written to ./data/vector_store/
```

### Query Flow

```
User asks: "What were the Q3 revenue figures?"
    │
    ▼
SentenceTransformer encodes query → 384-dim vector
    │
    ▼
FAISS searches for 10 nearest child vectors (L2 distance)
    │
    ▼
Results iterated in order:
  - Child #7 → parent_id "parent_report.pdf_1" → NEW → add parent text → 1 result
  - Child #3 → parent_id "parent_report.pdf_0" → NEW → add parent text → 2 results ✅ stop
    │
    ▼
Prompt assembled:
  [SYSTEM] You are a helpful assistant...
  [USER] Context:
    Source: report.pdf
    Content: <parent chunk 1, first 500 chars>

    Source: report.pdf
    Content: <parent chunk 0, first 500 chars>

  Question: What were the Q3 revenue figures?
    │
    ▼
Llama-3.2-1B-Instruct generates answer (GPU, max 256 tokens)
    │
    ▼
Response: { "answer": "Q3 revenue was $4.2M...", "sources": ["report.pdf"] }
```

---

## 12. File Structure

```
SLM Proj/
├── backend/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, endpoints, startup
│   ├── ingest.py        # Document extraction + Hierarchical chunking
│   ├── rag.py           # FAISS vector store + hierarchical retrieval
│   ├── inference.py     # SLM engine (llama.cpp, GPU offload)
│   └── utils.py         # Logger, timer, config constants
│
├── ui/
│   └── web/
│       └── index.html   # Single-page upload & chat UI
│
├── data/
│   └── vector_store/    # FAISS index (auto-created)
│
├── models/
│   └── slm/             # Downloaded GGUF model (auto-populated)
│
├── Dockerfile           # nvidia/cuda runtime base, pre-built wheels
├── docker-compose.yml   # GPU passthrough, volume mounts
├── requirements.txt     # Python dependencies
└── ARCHITECTURE.md      # This document
```

---

## 13. Design Decisions & Trade-offs

| Decision | Choice Made | Alternative Considered | Reason |
|---|---|---|---|
| **Chunking** | Hierarchical Parent-Child | Flat fixed-size chunks | Better retrieval precision + richer context for large docs |
| **Vector Index** | `FAISS IndexFlatL2` (exact) | HNSW / IVF (approximate) | No build overhead, zero approximation error at current scale |
| **SLM** | Llama-3.2-1B Q4_K_M | Qwen2.5-1.5B Q4_K_M | Smaller, faster, excellent instruction following, lower VRAM |
| **Inference Engine** | `llama-cpp-python` (GGUF) | HuggingFace Transformers | GGUF+llama.cpp is 3-5x faster and lower memory than PyTorch |
| **GPU Setup** | Pre-built CUDA wheels | Compile from source | Source compilation fails in Docker (no host libcuda.so at build time) |
| **Base Image** | `nvidia/cuda:runtime` | `nvidia/cuda:devel` | Runtime image is ~2GB smaller; devel only needed for compilation |
| **Temperature** | `0.1` | `0.7`+ | Low temperature = deterministic, factual, less hallucination |
| **Max Tokens** | `256` | `512`+ | Keeps latency low; most support answers fit in 256 tokens |
| **Context Limit** | `500 chars` per chunk in prompt | Full parent text | Balances context richness vs. SLM prompt length / latency |
| **Embedding Model** | `all-MiniLM-L6-v2` | `bge-base-en` / `e5-small` | Best speed/accuracy ratio, widely tested for RAG |

---

## 14. External API Integration

The backend is designed to be consumed as a REST API by any external application — website, mobile app, chatbot, internal tool, etc.

### Developer Documentation Page

A full interactive API reference is available at:

```
ui/web/api-docs.html
```

Open this file in a browser while the Docker container is running. It includes:
- Full endpoint reference with request/response schemas
- Copy-to-clipboard **code examples** in JavaScript, Python, and cURL
- A **React custom hook** example for embedding into web apps
- A **live playground** that sends real queries to your running server
- Server status indicator that auto-polls `/health`

### Base URL

All requests go to:
```
http://localhost:8000
```

To expose to other machines on your LAN, use your server's local IP address instead of `localhost`.

### CORS Policy

The API currently allows **all origins** (`*`). This is intentional for local/LAN use. If exposing publicly, restrict CORS in `backend/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-site.com"],  # restrict to your domain
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)
```

### Typical Integration Pattern

```
Your Website / App
    │
    ├─ POST /v1/documents/upload  ← admin uploads knowledge base docs
    │
    └─ POST /v1/query            ← end-user asks questions in real time
              │
              ▼
        { answer, sources }       ← display in your UI
```

### Minimal JavaScript Integration (copy-paste ready)

```javascript
async function askSLM(question) {
  const res = await fetch('http://localhost:8000/v1/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  const { answer, sources } = await res.json();
  return { answer, sources };
}
```

### Production Considerations

| Concern | Recommendation |
|---|---|
| **Authentication** | Add an `X-API-Key` header check in FastAPI middleware |
| **HTTPS** | Place Nginx or Caddy in front to terminate TLS |
| **Rate Limiting** | Add `slowapi` middleware to prevent query flooding |
| **Public Exposure** | Use Cloudflare Tunnel or Tailscale instead of opening ports |
| **Multi-user** | The current FAISS index is shared — one knowledge base for all users |
