import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from .utils import get_logger, STORAGE_PATH

logger = get_logger(__name__)

class RAGEngine:
    def __init__(self):
        # Using a small, fast model for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []  # Metadata storage
        self.index_file = os.path.join(STORAGE_PATH, "index.faiss")
        self.meta_file = os.path.join(STORAGE_PATH, "index.pkl")
        self.load_index()

    def add_documents(self, chunks: List[Dict]):
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts)
        
        self.index.add(np.array(embeddings).astype('float32'))
        self.documents.extend(chunks)
        
        self.save_index()
        logger.info(f"Added {len(chunks)} chunks to index. Total: {self.index.ntotal}")

    def search(self, query: str, k: int = 3) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        # Retrieve more child chunks to ensure we get enough unique parents
        search_k = max(k * 3, 10)
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), search_k)
        
        results = []
        seen_parents = set()
        
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = self.documents[idx]
                parent_id = doc.get("parent_id")
                
                # Hierarchical retrieval: return parent_text if available, deduplicate by parent_id
                if parent_id:
                    if parent_id not in seen_parents:
                        seen_parents.add(parent_id)
                        results.append({
                            "text": doc.get("parent_text", doc["text"]),
                            "source": doc["source"]
                        })
                else:
                    # Fallback for old flat chunks
                    if doc["text"] not in [r["text"] for r in results]:
                        results.append(doc)
                
                if len(results) >= k:
                    break
                    
        return results

    def save_index(self):
        if not os.path.exists(STORAGE_PATH):
            os.makedirs(STORAGE_PATH)
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.documents, f)

    def load_index(self):
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, "rb") as f:
                self.documents = pickle.load(f)
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
        else:
            logger.info("Initialized new empty index")

    def reset(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.save_index()
        logger.info("Index reset")
