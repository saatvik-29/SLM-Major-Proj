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

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "source": source
            })
            
        logger.info(f"Created {len(chunks)} chunks from {source}")
        return chunks
