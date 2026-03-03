import os
from typing import List
from fastapi import UploadFile
import pypdf
import docx
from .utils import get_logger

logger = get_logger(__name__)

class DocumentIngester:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def extract_text(self, file: UploadFile) -> str:
        filename = file.filename.lower()
        content = ""
        
        try:
            if filename.endswith('.pdf'):
                reader = pypdf.PdfReader(file.file)
                for page in reader.pages:
                    content += page.extract_text() + "\n"
            elif filename.endswith('.docx'):
                doc = docx.Document(file.file)
                content = "\n".join([para.text for para in doc.paragraphs])
            elif filename.endswith('.txt') or filename.endswith('.md'):
                content = (await file.read()).decode('utf-8')
            else:
                logger.warning(f"Unsupported file type: {filename}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {e}")
            return ""
            
        return self.clean_text(content)

    def clean_text(self, text: str) -> str:
        # Basic cleaning: remove excessive whitespace
        return " ".join(text.split())

    def chunk_text(self, text: str, source: str) -> List[dict]:
        words = text.split()
        chunks = []
        
        if not words:
            return []

        # Hierarchical Tree Approach: Large Parent Chunks -> Small Child Chunks
        parent_chunk_size = 800
        parent_chunk_overlap = 150
        
        child_chunk_size = 200
        child_chunk_overlap = 50

        parent_id_counter = 0
        for i in range(0, len(words), parent_chunk_size - parent_chunk_overlap):
            parent_chunk_words = words[i:i + parent_chunk_size]
            parent_text = " ".join(parent_chunk_words)
            parent_id = f"parent_{source}_{parent_id_counter}"
            parent_id_counter += 1
            
            for j in range(0, len(parent_chunk_words), child_chunk_size - child_chunk_overlap):
                child_words = parent_chunk_words[j:j + child_chunk_size]
                if not child_words:
                    continue
                child_text = " ".join(child_words)
                
                chunks.append({
                    "text": child_text,         # Used for Faiss embedding and searching
                    "parent_text": parent_text, # Full parent context for SLM generation
                    "source": source,
                    "parent_id": parent_id
                })
            
        logger.info(f"Created {len(chunks)} hierarchical child chunks from {source}")
        return chunks
