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
        # Preserve newlines so section headers stay intact.
        # Only collapse multiple blank lines into one.
        import re
        text = re.sub(r'\r\n', '\n', text)          # normalise Windows line endings
        text = re.sub(r'[ \t]+', ' ', text)          # collapse horizontal whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)       # max one blank line between sections
        return text.strip()

    def chunk_text(self, text: str, source: str) -> List[dict]:
        """
        Section-aware hierarchical chunking.

        For structured documents (with == SECTION == headers) each section
        is kept as its own parent chunk so information about one hostel block
        never bleeds into another block's chunk.

        For unstructured documents we fall back to the original sliding-window
        word-based parent→child approach.
        """
        import re
        chunks = []

        # ── 1. Try section-based splitting on == HEADER == markers ──────────
        sections = re.split(r'(?=^==\s+)', text, flags=re.MULTILINE)
        sections = [s.strip() for s in sections if s.strip()]

        if len(sections) > 1:
            # Structured document: one parent per section
            for idx, section in enumerate(sections):
                parent_id   = f"parent_{source}_{idx}"
                parent_text = section

                # Build fine-grained child chunks from this section for FAISS
                words = section.split()
                child_size    = 80   # words per child embedding
                child_overlap = 20

                for j in range(0, len(words), child_size - child_overlap):
                    child_words = words[j:j + child_size]
                    if not child_words:
                        continue
                    chunks.append({
                        "text":        " ".join(child_words),  # embedded in FAISS
                        "parent_text": parent_text,            # returned to LLM
                        "source":      source,
                        "parent_id":   parent_id,
                    })

            logger.info(f"Section-aware: {len(sections)} sections → {len(chunks)} child chunks from {source}")
            return chunks

        # ── 2. Fallback: sliding-window word-based chunking ──────────────────
        words = text.split()
        if not words:
            return []

        parent_chunk_size    = 800
        parent_chunk_overlap = 150
        child_chunk_size     = 200
        child_chunk_overlap  = 50

        parent_id_counter = 0
        for i in range(0, len(words), parent_chunk_size - parent_chunk_overlap):
            parent_chunk_words = words[i:i + parent_chunk_size]
            parent_text        = " ".join(parent_chunk_words)
            parent_id          = f"parent_{source}_{parent_id_counter}"
            parent_id_counter += 1

            for j in range(0, len(parent_chunk_words), child_chunk_size - child_chunk_overlap):
                child_words = parent_chunk_words[j:j + child_chunk_size]
                if not child_words:
                    continue
                chunks.append({
                    "text":        " ".join(child_words),
                    "parent_text": parent_text,
                    "source":      source,
                    "parent_id":   parent_id,
                })

        logger.info(f"Flat fallback: {len(chunks)} hierarchical child chunks from {source}")
        return chunks

