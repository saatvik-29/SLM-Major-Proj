import os
import requests
from llama_cpp import Llama
from .utils import get_logger, MODEL_PATH

logger = get_logger(__name__)

# Constants for the model
# Using Llama-3.2-3B-Instruct-Q4_K_M GGUF from bartowski
MODEL_URL = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MODEL_FILENAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

class SLMEngine:
    def __init__(self):
        self.model_path = os.path.join(MODEL_PATH, MODEL_FILENAME)
        self.ensure_model_exists()
        
        # Initialize Llama model
        # n_ctx should cover context + query + answer. 4096 is standard.
        # n_gpu_layers=-1 attempts to offload all to GPU if available.
        logger.info(f"Loading SLM from {self.model_path}...")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=4096,
            n_threads=os.cpu_count(),
            n_gpu_layers=-1, 
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
        # System prompt ensuring strict adherence to context
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful support assistant. Answer the user's question using ONLY the provided context below. "
                "If the answer is not present in the context, say \"I don't know based on the provided documents.\" "
                "Do NOT use outside knowledge."
            )
        }
        
        user_message = {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }

        # Use create_chat_completion for standardized templating
        output = self.llm.create_chat_completion(
            messages=[system_message, user_message],
            max_tokens=512,
            temperature=0.7,
        )
        
        return output['choices'][0]['message']['content'].strip()
