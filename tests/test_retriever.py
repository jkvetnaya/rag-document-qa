# ============================================================================
# FILE: tests/test_retriever.py
# ============================================================================
import pytest
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.retriever import Retriever
from src.embedding_system import EmbeddingSystem
from config import Config

class TestConfig(Config):
    """Test configuration with custom settings."""
    def __init__(self, temp_dir=None):
        super().__init__()
        if temp_dir:
            self.DATA_PATH = temp_dir
        self.TOP_K_CHUNKS = 3
        self.SIMILARITY_THRESHOLD = 0.1
        self.MAX_FEATURES = 100
        self.USE_OPENAI_EMBEDDINGS = False
        self.OPENAI_API_KEY = None

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    return TestConfig(temp_dir)

@pytest.fixture
def sample_chunks():
    """Sample chunks with various content for testing retrieval."""
    return [
        {
            'text': 'Machine learning is a subset of artificial intelligence that focuses on algorithms.',
            'source': 'ml_basics.txt',
            'chunk_index': 0,
            'total_chunks': 2
        },
        {
            'text': 'Deep learning uses neural networks with multiple layers to process data.',
            'source': 'ml_basics.txt',
            'chunk_index': 1,
            'total_chunks': 2
        },
        {
            'text': 'Natural language processing enables computers to understand human language.',
            'source': 'nlp_guide.txt',
            'chunk_index': 0,
            'total_chunks': 3
        },
        {
            'text': 'Vector embeddings represent text as numerical arrays in high-dimensional space.',
            'source': 'nlp_guide.txt',
            'chunk_index': 1,
            'total_chunks': 3
        },
        {
            'text': 'Similarity search finds the most relevant documents using cosine similarity.',
            'source': 'nlp_guide.txt',
            'chunk_index': 2,
            'total_chunks': 3
        },
        {
            'text': 'Python is a popular programming language for data science and machine learning.',
            'source': 'programming.txt',
            'chunk_index': 0,
            'total_chunks': 1
        }
    ]

@pytest.fixture
def mock_embedding_system_tfidf(test_config, sample_chunks):
    """Create a mock embedding system with TF-IDF embeddings."""
    embedding_system = Mock(spec=EmbeddingSystem)
    embedding_system.config = test_config
    embedding_system.use_openai = False
    embedding_system.chunks = sample_chunks
    
    # Create mock TF-IDF embeddings (sparse matrix simulation)
    num_chunks = len(sample_chunks)
    num_features = 50
    
    # Create realistic sparse embeddings
    embeddings = np.random.rand(num_chunks, num_features) * 0.3
    # Make embeddings more realistic by zeroing out most values
    mask = np.random.rand(num_chunks, num_features) < 0.8
    embeddings[mask] = 0
    
    # Create mock sparse matrix
    mock_sparse_matrix = Mock()
    mock_sparse_matrix.shape = (num_chunks, num_features)
    mock_sparse_matrix.toarray.return_value = embeddings
    
    embedding_system.embeddings = mock_sparse_matrix
    
    # Mock embed_query method
    def mock_embed_query(query):
        # Return query embedding that's similar to first chunk for "machine learning"
        query_embedding = np.random.rand(1, num_features) * 0.2
        if "machine learning" in query.lower():
            query_embedding[0, :10] = embeddings[0, :10] + 0.1  # Make similar to first chunk
        elif "neural networks" in query.lower():
            query_embedding[0, :10] = embeddings[1, :10] + 0.1  # Make similar to second chunk
        elif "language" in query.lower():
            query_embedding[0, :10] = embeddings[2, :10] + 0.1  # Make similar to NLP chunk
        
        # Create mock sparse matrix for query
        mock_query_sparse = Mock()
        mock_query_sparse.shape = (1, num_features)
        mock_query_sparse.toarray.return_value = query_embedding
        return mock_query_sparse
    
    embedding_system.embed_query = mock_embed_query
    return embedding_system

@pytest.fixture
def mock_embedding_system_openai(test_config, sample_chunks):
    """Create a mock embedding system with OpenAI embeddings."""
    embedding_system = Mock(spec=EmbeddingSystem)
    embedding_system.config = test_config
    embedding_system.use_openai = True
    embedding_system.chunks = sample_chunks
    
    # Create mock dense embeddings
    num_chunks = len(sample_chunks)
    embedding_dim = 1536  # OpenAI embedding dimension
    
    # Create realistic dense embeddings
    embeddings = np.random.randn(num_chunks, embedding_dim) * 0.1
    # Normalize to unit vectors (typical for OpenAI embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    embedding_system.embeddings = embeddings
    
    # Mock embed_query method
    def mock_embed_query(query):
        query_embedding = np.random.randn(1, embedding_dim) * 0.1
        # Make query similar to relevant chunks
        if "machine learning" in query.lower():
            query_embedding = embeddings[0:1] + np.random.randn(1, embedding_dim) * 0.05
        elif "neural networks" in query.lower():
            query_embedding = embeddings[1:2] + np.random.randn(1, embedding_dim) * 0.05
        elif "language" in query.lower():
            query_embedding = embeddings[2:3] + np.random.randn(1, embedding_dim) * 0.05
        
        # Normalize
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        return query_embedding
    
    embedding_system.embed_query = mock_embed_query
    return embedding_system

@pytest.fixture
def retriever_tfidf(mock_embedding_system_tfidf, test_config):
    """Create a retriever with TF-IDF embedding system."""
    return Retriever(mock_embedding_system_tfidf, test_config)

@pytest.fixture
def retriever_openai(mock_embedding_system_openai, test_config):
    """Create a retriever with OpenAI embedding system."""
    return Retriever(mock_embedding_system_openai, test_config)

class TestRetrieverInitialization:
    """Test retriever initialization."""
    
    def test_init_with_embedding_system_and_config(self, mock_embedding_system_tfidf, test_config):
        """Test initialization with embedding system and config."""
        retriever = Retriever(mock_embedding_system_tfidf, test_config)
        
        assert retriever.embedding_system == mock_embedding_system_tfidf
        assert retriever.config == test_config
    
    def test_init_with_embedding_system_no_config(self, mock_embedding_system_tfidf):
        """Test initialization with embedding system but no config."""
        retriever = Retriever(mock_embedding_system_tfidf)
        
        assert retriever.embedding_system == mock_embedding_system_tfidf
        assert retriever.config is not None
        assert isinstance(retriever.config, Config)
    
    def test_config_values_used(self, mock_embedding_system_tfidf, test_config):
        """Test that config values are properly used."""
        retriever = Retriever(mock_embedding_system_tfidf, test_config)
        
        assert retriever.config.TOP_K_CHUNKS == 3
        assert retriever.config.SIMILARITY_THRESHOLD == 0.1

class TestRetrieveRelevantChunks:
    """Test chunk retrieval functionality."""
    
    def test_retrieve_relevant_chunks_tfidf_basic(self, retriever_tfidf):
        """Test basic chunk retrieval with TF-IDF embeddings."""
        query = "machine learning algorithms"
        
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            # Mock similarity scores
            mock_cosine.return_value.flatten.return_value = np.array([0.8, 0.3, 0.2, 0.1, 0.05, 0.4])
            
            chunks = retriever_tfidf.retrieve_relevant_chunks(query)
        
        # Should return top 3 chunks (TOP_K_CHUNKS = 3)
        assert len(chunks) <= 3
        
        # Check that chunks have similarity scores
        for chunk in chunks:
            assert 'similarity_score' in chunk
            assert chunk['similarity_score'] >= retriever_tfidf.config.SIMILARITY_THRESHOLD
        
        # Should be sorted by similarity (highest first)
        scores = [chunk['similarity_score'] for chunk in chunks]
        assert scores == sorted(scores, reverse=True)
    
    def test_retrieve_relevant_chunks_openai_basic(self, retriever_openai):
        """Test basic chunk retrieval with OpenAI embeddings."""
        query = "neural networks deep learning"
        
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            # Mock similarity scores for dense vectors
            mock_cosine.return_value = np.array([[0.9, 0.7, 0.3, 0.2, 0.1, 0.5]])
            
            chunks = retriever_openai.retrieve_relevant_chunks(query)
        
        assert len(chunks) <= 3
        
        # Check similarity scores
        for chunk in chunks:
            assert 'similarity_score' in chunk
            assert chunk['similarity_score'] >= retriever_openai.config.SIMILARITY_THRESHOLD
    
    def test_retrieve_relevant_chunks_custom_top_k(self, retriever_tfidf):
        """Test retrieval with custom top_k parameter."""
        query = "artificial intelligence"
        custom_top_k = 2
        
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            mock_cosine.return_value.flatten.return_value = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
            
            chunks = retriever_tfidf.retrieve_relevant_chunks(query, top_k=custom_top_k)
        
        assert len(chunks) <= custom_top_k
    
    def test_retrieve_relevant_chunks_similarity_threshold_filter(self, retriever_tfidf):
        """Test that chunks below similarity threshold are filtered out."""
        query = "unrelated topic"
        
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            # All similarities below threshold
            mock_cosine.return_value.flatten.return_value = np.array([0.05, 0.03, 0.02, 0.01, 0.01, 0.04])
            
            chunks = retriever_tfidf.retrieve_relevant_chunks(query)
        
        # Should return empty list since all similarities below threshold (0.1)
        assert len(chunks) == 0
    
    def test_retrieve_relevant_chunks_no_embeddings_error(self, test_config):
        """Test error when no embeddings are available."""
        # Create embedding system without embeddings
        embedding_system = Mock()
        embedding_system.embeddings = None
        
        retriever = Retriever(embedding_system, test_config)
        
        with pytest.raises(ValueError, match="No embeddings found"):
            retriever.retrieve_relevant_chunks("test query")
    
    def test_retrieve_relevant_chunks_preserves_original_metadata(self, retriever_tfidf, sample_chunks):
        """Test that original chunk metadata is preserved."""
        query = "machine learning"
        
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            mock_cosine.return_value.flatten.return_value = np.array([0.8, 0.3, 0.2, 0.1, 0.05, 0.4])
            
            chunks = retriever_tfidf.retrieve_relevant_chunks(query)
        
        # Check that original metadata is preserved
        for chunk in chunks:
            assert 'text' in chunk
            assert 'source' in chunk
            assert 'chunk_index' in chunk
            assert 'total_chunks' in chunk
            assert 'similarity_score' in chunk  # Added by retriever
            
            # Verify metadata matches original chunks
            original_chunk = next(
                c for c in sample_chunks 
                if c['text'] == chunk['text'] and c['source'] == chunk['source']
            )
            assert chunk['chunk_index'] == original_chunk['chunk_index']
            assert chunk['total_chunks'] == original_chunk['total_chunks']
    
    def test_retrieve_relevant_chunks_empty_query(self, retriever_tfidf):
        """Test retrieval with empty query."""
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            mock_cosine.return_value.flatten.return_value = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
            
            chunks = retriever_tfidf.retrieve_relevant_chunks("")
        
        # Should still work, even with empty query
        assert isinstance(chunks, list)

class TestFormatContext:
    """Test context formatting functionality."""
    
    def test_format_context_multiple_chunks(self, retriever_tfidf):
        """Test formatting multiple chunks into context."""
        chunks = [
            {
                'text': 'First relevant chunk about machine learning.',
                'source': 'doc1.txt',
                'similarity_score': 0.9
            },
            {
                'text': 'Second relevant chunk about neural networks.',
                'source': 'doc2.txt',
                'similarity_score': 0.7
            },
            {
                'text': 'Third relevant chunk about data science.',
                'source': 'doc1.txt',
                'similarity_score': 0.5
            }
        ]
        
        context = retriever_tfidf.format_context(chunks)
        
        # Check that all chunks are included
        assert 'machine learning' in context
        assert 'neural networks' in context
        assert 'data science' in context
        
        # Check that source information is included
        assert 'doc1.txt' in context
        assert 'doc2.txt' in context
        
        # Check that similarity scores are included
        assert '0.900' in context
        assert '0.700' in context
        assert '0.500' in context
        
        # Check formatting structure
        assert '[Context 1' in context
        assert '[Context 2' in context
        assert '[Context 3' in context
    
    def test_format_context_single_chunk(self, retriever_tfidf):
        """Test formatting single chunk into context."""
        chunks = [
            {
                'text': 'Single relevant chunk.',
                'source': 'single.txt',
                'similarity_score': 0.8
            }
        ]
        
        context = retriever_tfidf.format_context(chunks)
        
        assert 'Single relevant chunk' in context
        assert 'single.txt' in context
        assert '0.800' in context
        assert '[Context 1' in context
        assert '[Context 2' not in context
    
    def test_format_context_empty_chunks(self, retriever_tfidf):
        """Test formatting with empty chunks list."""
        context = retriever_tfidf.format_context([])
        
        assert context == "No relevant context found."
    
    def test_format_context_chunks_without_similarity_score(self, retriever_tfidf):
        """Test formatting chunks that don't have similarity scores."""
        chunks = [
            {
                'text': 'Chunk without similarity score.',
                'source': 'test.txt'
            }
        ]
        
        context = retriever_tfidf.format_context(chunks)
        
        # Should handle missing similarity score gracefully
        assert 'Chunk without similarity score' in context
        assert 'test.txt' in context
        assert '0.000' in context  # Default score
    
    def test_format_context_long_text_chunks(self, retriever_tfidf):
        """Test formatting with very long text chunks."""
        long_text = "Very long chunk text. " * 50
        chunks = [
            {
                'text': long_text,
                'source': 'long.txt',
                'similarity_score': 0.6
            }
        ]
        
        context = retriever_tfidf.format_context(chunks)
        
        # Should include the full text
        assert long_text in context
        assert 'long.txt' in context
    
    def test_format_context_special_characters(self, retriever_tfidf):
        """Test formatting chunks with special characters."""
        chunks = [
            {
                'text': 'Text with émojis 🚀 and spéciał chåractërs!',
                'source': 'special.txt',
                'similarity_score': 0.7
            }
        ]
        
        context = retriever_tfidf.format_context(chunks)
        
        assert 'émojis 🚀' in context
        assert 'spéciał chåractërs' in context
        assert 'special.txt' in context

class TestRetrieverIntegration:
    """Integration tests combining retrieval and formatting."""
    
    def test_retrieve_and_format_workflow(self, retriever_tfidf):
        """Test complete retrieve and format workflow."""
        query = "machine learning and AI"
        
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            mock_cosine.return_value.flatten.return_value = np.array([0.9, 0.3, 0.7, 0.2, 0.1, 0.5])
            
            # Retrieve chunks
            chunks = retriever_tfidf.retrieve_relevant_chunks(query)
            
            # Format context
            context = retriever_tfidf.format_context(chunks)
        
        assert len(chunks) > 0
        assert isinstance(context, str)
        assert len(context) > 0
        assert "No relevant context found" not in context
    
    def test_retrieve_and_format_no_relevant_chunks(self, retriever_tfidf):
        """Test workflow when no relevant chunks are found."""
        query = "completely unrelated topic"
        
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            # All similarities below threshold
            mock_cosine.return_value.flatten.return_value = np.array([0.05, 0.03, 0.02, 0.01, 0.01, 0.04])
            
            chunks = retriever_tfidf.retrieve_relevant_chunks(query)
            context = retriever_tfidf.format_context(chunks)
        
        assert len(chunks) == 0
        assert context == "No relevant context found."
    
    def test_different_embedding_systems_consistency(self, retriever_tfidf, retriever_openai):
        """Test that both TF-IDF and OpenAI retrievers return consistent structure."""
        query = "natural language processing"
        
        # Test TF-IDF retriever
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            mock_cosine.return_value.flatten.return_value = np.array([0.2, 0.3, 0.8, 0.7, 0.1, 0.4])
            tfidf_chunks = retriever_tfidf.retrieve_relevant_chunks(query)
        
        # Test OpenAI retriever
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            mock_cosine.return_value = np.array([[0.2, 0.3, 0.8, 0.7, 0.1, 0.4]])
            openai_chunks = retriever_openai.retrieve_relevant_chunks(query)
        
        # Both should return same structure
        for chunks in [tfidf_chunks, openai_chunks]:
            for chunk in chunks:
                assert 'text' in chunk
                assert 'source' in chunk
                assert 'chunk_index' in chunk
                assert 'total_chunks' in chunk
                assert 'similarity_score' in chunk

class TestRetrieverEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_retrieve_with_nan_similarities(self, retriever_tfidf):
        """Test handling of NaN similarity scores."""
        query = "test query"
        
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            # Include NaN values in similarities
            mock_cosine.return_value.flatten.return_value = np.array([0.8, np.nan, 0.6, 0.4, np.nan, 0.2])
            
            chunks = retriever_tfidf.retrieve_relevant_chunks(query)
        
        # Should handle NaN gracefully
        assert isinstance(chunks, list)
        # All returned chunks should have valid similarity scores
        for chunk in chunks:
            assert not np.isnan(chunk['similarity_score'])
    
    def test_retrieve_with_infinite_similarities(self, retriever_tfidf):
        """Test handling of infinite similarity scores."""
        query = "test query"
        
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            mock_cosine.return_value.flatten.return_value = np.array([0.8, np.inf, 0.6, 0.4, -np.inf, 0.2])
            
            chunks = retriever_tfidf.retrieve_relevant_chunks(query)
        
        # Should handle infinite values gracefully
        assert isinstance(chunks, list)
        for chunk in chunks:
            assert np.isfinite(chunk['similarity_score'])

    def test_retrieve_with_very_high_top_k(self, retriever_tfidf):
        """Test retrieval with top_k larger than number of chunks."""
        query = "test query"
        very_high_k = 1000
        
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            mock_cosine.return_value.flatten.return_value = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
            
            chunks = retriever_tfidf.retrieve_relevant_chunks(query, top_k=very_high_k)
        
        # Should return all available chunks (6 in sample_chunks)
        assert len(chunks) <= 6
    
    def test_retrieve_with_zero_top_k(self, retriever_tfidf):
        """Test retrieval with top_k = 0."""
        query = "test query"
        
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            mock_cosine.return_value.flatten.return_value = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
            
            chunks = retriever_tfidf.retrieve_relevant_chunks(query, top_k=0)
        
        # Should return empty list
        assert len(chunks) == 0
    
    def test_retrieve_with_negative_similarities(self, retriever_tfidf):
        """Test handling of negative similarity scores."""
        query = "test query"
        
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            mock_cosine.return_value.flatten.return_value = np.array([0.8, -0.2, 0.6, -0.5, 0.4, 0.3])
            
            chunks = retriever_tfidf.retrieve_relevant_chunks(query)
        
        # Should handle negative similarities
        assert isinstance(chunks, list)
        # Only positive similarities above threshold should be returned
        for chunk in chunks:
            assert chunk['similarity_score'] >= retriever_tfidf.config.SIMILARITY_THRESHOLD

class TestRetrieverPerformance:
    """Test performance-related aspects."""
    
    def test_retrieve_with_large_number_of_chunks(self, test_config):
        """Test retrieval performance with large number of chunks."""
        # Create large number of chunks
        large_chunks = []
        for i in range(1000):
            large_chunks.append({
                'text': f'Chunk {i} with various content about topic {i % 10}.',
                'source': f'doc_{i // 100}.txt',
                'chunk_index': i % 100,
                'total_chunks': 100
            })
        
        # Create mock embedding system
        embedding_system = Mock()
        embedding_system.use_openai = False
        embedding_system.chunks = large_chunks
        embedding_system.embeddings = Mock()
        embedding_system.embeddings.shape = (1000, 100)
        
        # Mock embed_query and cosine_similarity
        embedding_system.embed_query.return_value = Mock()
        
        retriever = Retriever(embedding_system, test_config)
        
        with patch('src.retriever.cosine_similarity') as mock_cosine:
            # Create similarities for 1000 chunks
            similarities = np.random.rand(1000) * 0.5 + 0.1  # Range 0.1 to 0.6
            mock_cosine.return_value.flatten.return_value = similarities
            
            chunks = retriever.retrieve_relevant_chunks("test query")
        
        # Should return top_k chunks efficiently
        assert len(chunks) <= test_config.TOP_K_CHUNKS
        assert len(chunks) > 0

# Utility functions for testing
def create_mock_sparse_matrix(data):
    """Create a mock sparse matrix for TF-IDF testing."""
    mock_matrix = Mock()
    mock_matrix.shape = data.shape
    mock_matrix.toarray.return_value = data
    return mock_matrix

def assert_valid_chunk_structure(chunk):
    """Assert that a chunk has the expected structure."""
    required_fields = ['text', 'source', 'chunk_index', 'total_chunks']
    for field in required_fields:
        assert field in chunk
    
    assert isinstance(chunk['text'], str)
    assert isinstance(chunk['source'], str)
    assert isinstance(chunk['chunk_index'], int)
    assert isinstance(chunk['total_chunks'], int)
    assert chunk['chunk_index'] < chunk['total_chunks']

if __name__ == "__main__":
    pytest.main([__file__, "-v"])