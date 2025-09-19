# ============================================================================
# FILE: src/embedding_system.py
# ============================================================================
import numpy as np
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from config import Config

# Optional OpenAI import
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class EmbeddingSystem:
    """Handles text embedding creation and storage."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.vectorizer = None
        self.embeddings = None
        self.chunks = None
        self.use_openai = (self.config.USE_OPENAI_EMBEDDINGS and 
                          OPENAI_AVAILABLE and 
                          self.config.OPENAI_API_KEY)
        
        if self.use_openai:
            openai.api_key = self.config.OPENAI_API_KEY
            print("Using OpenAI embeddings")
        else:
            print("Using TF-IDF embeddings")
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Create embeddings for text chunks."""
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        if self.use_openai:
            return self._create_openai_embeddings(texts)
        else:
            return self._create_tfidf_embeddings(texts)
    
    def _create_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create TF-IDF embeddings."""
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.MAX_FEATURES,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.embeddings = self.vectorizer.fit_transform(texts)
        print(f"Created TF-IDF embeddings: {self.embeddings.shape}")
        return self.embeddings
    
    def _create_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create OpenAI embeddings."""
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                response = openai.embeddings.create(
                    model=self.config.EMBEDDING_MODEL,
                    input=text
                )
                embeddings.append(response.data[0].embedding)
                
                if (i + 1) % 10 == 0:
                    print(f"Created embeddings for {i + 1}/{len(texts)} chunks")
                    
            except Exception as e:
                print(f"Error creating embedding for chunk {i}: {e}")
                # Fallback to zero vector
                embeddings.append([0.0] * 1536)  # Default embedding size
        
        self.embeddings = np.array(embeddings)
        print(f"Created OpenAI embeddings: {self.embeddings.shape}")
        return self.embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Create embedding for a query."""
        if self.use_openai:
            try:
                response = openai.embeddings.create(
                    model=self.config.EMBEDDING_MODEL,
                    input=query
                )
                return np.array([response.data[0].embedding])
            except Exception as e:
                print(f"Error creating query embedding: {e}")
                return np.zeros((1, 1536))
        else:
            if self.vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted")
            return self.vectorizer.transform([query])
    
    def save_embeddings(self, filepath: str = None):
        """Save embeddings and vectorizer to disk."""
        if filepath is None:
            filepath = os.path.join(self.config.DATA_PATH, "embeddings.pkl")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'embeddings': self.embeddings,
            'chunks': self.chunks,
            'vectorizer': self.vectorizer,
            'use_openai': self.use_openai
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str = None):
        """Load embeddings and vectorizer from disk."""
        if filepath is None:
            filepath = os.path.join(self.config.DATA_PATH, "embeddings.pkl")
        
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.embeddings = data['embeddings']
            self.chunks = data['chunks']
            self.vectorizer = data.get('vectorizer')
            self.use_openai = data.get('use_openai', False)
            
            print(f"Loaded embeddings from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False