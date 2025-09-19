# ============================================================================
# FILE: src/retriever.py
# ============================================================================
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from src.embedding_system import EmbeddingSystem
from config import Config

class Retriever:
    """Handles similarity search and chunk retrieval."""
    
    def __init__(self, embedding_system: EmbeddingSystem, config: Config = None):
        self.embedding_system = embedding_system
        self.config = config or Config()
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve most relevant chunks for a query."""
        top_k = top_k or self.config.TOP_K_CHUNKS
        
        if self.embedding_system.embeddings is None:
            raise ValueError("No embeddings found. Please create embeddings first.")
        
        # Get query embedding
        query_embedding = self.embedding_system.embed_query(query)
        
        # Calculate similarities
        if self.embedding_system.use_openai:
            # For dense vectors (OpenAI)
            similarities = cosine_similarity(
                query_embedding, 
                self.embedding_system.embeddings
            )[0]
        else:
            # For sparse vectors (TF-IDF)
            similarities = cosine_similarity(
                query_embedding, 
                self.embedding_system.embeddings
            ).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by similarity threshold
        relevant_chunks = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= self.config.SIMILARITY_THRESHOLD:
                chunk = self.embedding_system.chunks[idx].copy()
                chunk['similarity_score'] = float(similarity)
                relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string."""
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk['source']
            text = chunk['text']
            score = chunk.get('similarity_score', 0)
            
            context_parts.append(
                f"[Context {i} from {source} (relevance: {score:.3f})]:\n{text}\n"
            )
        
        return "\n".join(context_parts)