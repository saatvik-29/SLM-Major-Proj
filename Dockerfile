FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
# build-essential and cmake are required for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install dependencies
# Set CMAKE_ARGS to enable CPU optimizations if needed, but defaults are usually fine for generic support
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend backend/
COPY ui ui/

# Create data directories
RUN mkdir -p data/vector_store models/slm

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
