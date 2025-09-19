# ============================================================================
# FILE: src/document_processor.py
# ============================================================================
import os
import re
from typing import List, Dict, Tuple
from config import Config

class DocumentProcessor:
    """Handles document loading and chunking operations."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def load_documents(self, folder_path: str = None) -> Dict[str, str]:
        """Load all .txt files from the specified folder."""
        folder_path = folder_path or self.config.DOCUMENTS_PATH
        documents = {}
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created documents folder: {folder_path}")
            return documents
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(folder_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        documents[filename] = file.read()
                    print(f"Loaded document: {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return documents
    
    def chunk_text(self, text: str, chunk_size: int = None, 
                   overlap: int = None) -> List[str]:
        """Split text into overlapping chunks."""
        chunk_size = chunk_size or self.config.CHUNK_SIZE
        overlap = overlap or self.config.CHUNK_OVERLAP
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text.strip())
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If we're not at the end, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_documents(self, documents: Dict[str, str]) -> List[Dict]:
        """Process all documents and return chunks with metadata."""
        all_chunks = []
        
        for doc_name, content in documents.items():
            chunks = self.chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    'text': chunk,
                    'source': doc_name,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
                all_chunks.append(chunk_data)
        
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
