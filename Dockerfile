FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

COPY requirements.txt .

# Install llama-cpp-python from pre-built CUDA 12.1 wheel (no compilation needed)
RUN pip install --no-cache-dir \
    llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Install remaining dependencies
RUN pip install --no-cache-dir fastapi uvicorn python-multipart sentence-transformers faiss-cpu pypdf python-docx requests

COPY backend backend/
COPY ui ui/

RUN mkdir -p data/vector_store models/slm

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
