# ============================================================================
# FILE: config.py
# ============================================================================
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Document processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    DOCUMENTS_PATH = "documents"
    DATA_PATH = "data"
    
    # Embedding settings
    EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI model
    USE_OPENAI_EMBEDDINGS = True  # Set to True if you have OpenAI API key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Retrieval settings
    TOP_K_CHUNKS = 5 
    SIMILARITY_THRESHOLD = 0.1
    
    # TF-IDF settings (fallback)
    MAX_FEATURES = 1000
