import os
import requests
from llama_cpp import Llama
from .utils import get_logger, MODEL_PATH

logger = get_logger(__name__)

# Constants for the model
# Using Llama-3.2-1B-Instruct-Q4_K_M GGUF for fast GPU inference
MODEL_URL = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
MODEL_FILENAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
class SLMEngine:
    def __init__(self):
        self.model_path = os.path.join(MODEL_PATH, MODEL_FILENAME)
        self.ensure_model_exists()
        
        # Initialize Llama model with optimized parameters for low latency
        # n_ctx=2048: reduced context window (sufficient for 2 chunks + question + answer)
        # n_batch=512: explicit prompt processing batch size
        # flash_attn=True: faster attention computation
        logger.info(f"Loading SLM from {self.model_path}...")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_threads=os.cpu_count(),
            n_batch=512,
            n_gpu_layers=-1,
            flash_attn=True,
            verbose=False
        )
        logger.info("SLM loaded successfully.")

    def ensure_model_exists(self):
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
            
        if not os.path.exists(self.model_path):
            logger.info(f"Model not found. Downloading from {MODEL_URL}...")
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()
                with open(self.model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info("Model downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                raise RuntimeError("Could not download model.")

    def generate_answer(self, context: str, question: str) -> str:
        # System prompt: concise answers reduce generation time
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful support assistant. Answer the user's question "
                "using ONLY the provided context. Be concise and direct. "
                "If the answer is not in the context, say "
                "\"I don't know based on the provided documents.\""
            )
        }
        
        user_message = {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }

        # Optimized generation parameters:
        # max_tokens=256: cap output length for faster responses
        # temperature=0.1: near-greedy for factual Q&A (faster convergence)
        output = self.llm.create_chat_completion(
            messages=[system_message, user_message],
            max_tokens=256,
            temperature=0.1,
        )
        
        return output['choices'][0]['message']['content'].strip()
