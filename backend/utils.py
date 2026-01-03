import logging
import os
from contextlib import contextmanager
import time

# Configuration
STORAGE_PATH = "data/vector_store"
MODEL_PATH = "models/slm"
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md'}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def get_logger(name: str):
    return logging.getLogger(name)

logger = get_logger("slm_support")

@contextmanager
def timer(description: str):
    start = time.time()
    yield
    elapsed = time.time() - start
    logger.info(f"{description} took {elapsed:.2f} seconds")
