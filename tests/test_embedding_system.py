# ============================================================================
# FILE: tests/test_embedding_system.py
# ============================================================================
import pytest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.embedding_system import EmbeddingSystem
from config import Config

class TestConfig(Config):
    """Test configuration with temporary paths."""
    def __init__(self, temp_dir):
        super().__init__()
        self.DATA_PATH = temp_dir
        self.MAX_FEATURES = 100  # Smaller for testing
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
    """Sample chunks for testing."""
    return [
        {
            'text': 'This is a sample document about machine learning.',
            'source': 'doc1.txt',
            'chunk_index': 0,
            'total_chunks': 2
        },
        {
            'text': 'Natural language processing is a subset of AI.',
            'source': 'doc1.txt',
            'chunk_index': 1,
            'total_chunks': 2
        },
        {
            'text': 'Vector embeddings represent text as numerical arrays.',
            'source': 'doc2.txt',
            'chunk_index': 0,
            'total_chunks': 1
        }
    ]

@pytest.fixture
def embedding_system(test_config):
    """Create an embedding system instance."""
    return EmbeddingSystem(test_config)

class TestEmbeddingSystemInitialization:
    """Test embedding system initialization."""
    
    def test_init_with_config(self, test_config):
        """Test initialization with custom config."""
        system = EmbeddingSystem(test_config)
        assert system.config == test_config
        assert system.vectorizer is None
        assert system.embeddings is None
        assert system.chunks is None
        assert not system.use_openai
    
    def test_init_without_config(self):
        """Test initialization with default config."""
        system = EmbeddingSystem()
        assert system.config is not None
        assert isinstance(system.config, Config)
    
    def test_openai_availability_detection(self, test_config):
        """Test OpenAI availability detection."""
        # Test with OpenAI disabled
        test_config.USE_OPENAI_EMBEDDINGS = False
        system = EmbeddingSystem(test_config)
        assert not system.use_openai
        
        # Test with OpenAI enabled but no API key
        test_config.USE_OPENAI_EMBEDDINGS = True
        test_config.OPENAI_API_KEY = None
        system = EmbeddingSystem(test_config)
        assert not system.use_openai

class TestTFIDFEmbeddings:
    """Test TF-IDF embedding functionality."""
    
    def test_create_tfidf_embeddings(self, embedding_system, sample_chunks):
        """Test TF-IDF embedding creation."""
        embeddings = embedding_system.create_embeddings(sample_chunks)
        
        # Check that embeddings were created
        assert embeddings is not None
        assert embedding_system.embeddings is not None
        assert embedding_system.chunks == sample_chunks
        assert embedding_system.vectorizer is not None
        
        # Check dimensions
        assert embeddings.shape[0] == len(sample_chunks)
        assert embeddings.shape[1] <= embedding_system.config.MAX_FEATURES
    
    def test_tfidf_vectorizer_properties(self, embedding_system, sample_chunks):
        """Test TF-IDF vectorizer configuration."""
        embedding_system.create_embeddings(sample_chunks)
        
        vectorizer = embedding_system.vectorizer
        assert vectorizer.max_features == embedding_system.config.MAX_FEATURES
        assert vectorizer.stop_words == 'english'
        assert vectorizer.ngram_range == (1, 2)
    
    def test_embed_query_tfidf(self, embedding_system, sample_chunks):
        """Test query embedding with TF-IDF."""
        # First create embeddings
        embedding_system.create_embeddings(sample_chunks)
        
        # Test query embedding
        query = "machine learning algorithms"
        query_embedding = embedding_system.embed_query(query)
        
        assert query_embedding is not None
        assert query_embedding.shape[0] == 1
        assert query_embedding.shape[1] == embedding_system.embeddings.shape[1]
    
    def test_embed_query_without_fitted_vectorizer(self, embedding_system):
        """Test query embedding without fitted vectorizer raises error."""
        with pytest.raises(ValueError, match="TF-IDF vectorizer not fitted"):
            embedding_system.embed_query("test query")

class TestOpenAIEmbeddings:
    """Test OpenAI embedding functionality."""
    
    @patch('src.embedding_system.openai')
    def test_create_openai_embeddings_success(self, mock_openai, test_config, sample_chunks):
        """Test successful OpenAI embedding creation."""
        # Setup config for OpenAI
        test_config.USE_OPENAI_EMBEDDINGS = True
        test_config.OPENAI_API_KEY = "test_key"
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_openai.embeddings.create.return_value = mock_response
        
        # Patch OPENAI_AVAILABLE
        with patch('src.embedding_system.OPENAI_AVAILABLE', True):
            system = EmbeddingSystem(test_config)
            embeddings = system.create_embeddings(sample_chunks)
        
        # Verify results
        assert embeddings is not None
        assert embeddings.shape == (len(sample_chunks), 1536)
        assert system.use_openai
        assert mock_openai.embeddings.create.call_count == len(sample_chunks)
    
    @patch('src.embedding_system.openai')
    def test_create_openai_embeddings_api_error(self, mock_openai, test_config, sample_chunks):
        """Test OpenAI embedding creation with API error."""
        # Setup config for OpenAI
        test_config.USE_OPENAI_EMBEDDINGS = True
        test_config.OPENAI_API_KEY = "test_key"
        
        # Mock OpenAI to raise exception
        mock_openai.embeddings.create.side_effect = Exception("API Error")
        
        with patch('src.embedding_system.OPENAI_AVAILABLE', True):
            system = EmbeddingSystem(test_config)
            embeddings = system.create_embeddings(sample_chunks)
        
        # Should fallback to zero vectors
        assert embeddings is not None
        assert embeddings.shape == (len(sample_chunks), 1536)
        assert np.all(embeddings == 0)
    
    @patch('src.embedding_system.openai')
    def test_embed_query_openai_success(self, mock_openai, test_config, sample_chunks):
        """Test successful OpenAI query embedding."""
        # Setup config for OpenAI
        test_config.USE_OPENAI_EMBEDDINGS = True
        test_config.OPENAI_API_KEY = "test_key"
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 1536)]
        mock_openai.embeddings.create.return_value = mock_response
        
        with patch('src.embedding_system.OPENAI_AVAILABLE', True):
            system = EmbeddingSystem(test_config)
            system.create_embeddings(sample_chunks)
            
            query_embedding = system.embed_query("test query")
        
        assert query_embedding is not None
        assert query_embedding.shape == (1, 1536)
    
    @patch('src.embedding_system.openai')
    def test_embed_query_openai_error(self, mock_openai, test_config, sample_chunks):
        """Test OpenAI query embedding with API error."""
        # Setup config for OpenAI
        test_config.USE_OPENAI_EMBEDDINGS = True
        test_config.OPENAI_API_KEY = "test_key"
        
        # Mock OpenAI to work for create_embeddings but fail for query
        def side_effect(*args, **kwargs):
            if kwargs.get('input') == "test query":
                raise Exception("API Error")
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536)]
            return mock_response
        
        mock_openai.embeddings.create.side_effect = side_effect
        
        with patch('src.embedding_system.OPENAI_AVAILABLE', True):
            system = EmbeddingSystem(test_config)
            system.create_embeddings(sample_chunks)
            
            query_embedding = system.embed_query("test query")
        
        # Should return zero vector
        assert query_embedding is not None
        assert query_embedding.shape == (1, 1536)
        assert np.all(query_embedding == 0)

class TestEmbeddingPersistence:
    """Test embedding save/load functionality."""
    
    def test_save_embeddings(self, embedding_system, sample_chunks, temp_dir):
        """Test saving embeddings to disk."""
        # Create embeddings
        embedding_system.create_embeddings(sample_chunks)
        
        # Save embeddings
        filepath = os.path.join(temp_dir, "test_embeddings.pkl")
        embedding_system.save_embeddings(filepath)
        
        # Check file exists
        assert os.path.exists(filepath)
    
    def test_save_embeddings_default_path(self, embedding_system, sample_chunks):
        """Test saving embeddings with default path."""
        embedding_system.create_embeddings(sample_chunks)
        
        # Save with default path
        embedding_system.save_embeddings()
        
        # Check file exists in expected location
        expected_path = os.path.join(embedding_system.config.DATA_PATH, "embeddings.pkl")
        assert os.path.exists(expected_path)
    
    def test_load_embeddings_success(self, embedding_system, sample_chunks, temp_dir):
        """Test successfully loading embeddings."""
        # Create and save embeddings
        original_embeddings = embedding_system.create_embeddings(sample_chunks)
        filepath = os.path.join(temp_dir, "test_embeddings.pkl")
        embedding_system.save_embeddings(filepath)
        
        # Create new system and load embeddings
        new_system = EmbeddingSystem(embedding_system.config)
        result = new_system.load_embeddings(filepath)
        
        assert result is True
        assert new_system.chunks == sample_chunks
        assert new_system.vectorizer is not None
        np.testing.assert_array_equal(
            new_system.embeddings.toarray(), 
            original_embeddings.toarray()
        )
    
    def test_load_embeddings_file_not_found(self, embedding_system):
        """Test loading embeddings when file doesn't exist."""
        result = embedding_system.load_embeddings("nonexistent.pkl")
        assert result is False
    
    def test_load_embeddings_corrupted_file(self, embedding_system, temp_dir):
        """Test loading corrupted embeddings file."""
        # Create corrupted file
        filepath = os.path.join(temp_dir, "corrupted.pkl")
        with open(filepath, 'w') as f:
            f.write("corrupted data")
        
        result = embedding_system.load_embeddings(filepath)
        assert result is False

class TestEmbeddingSystemIntegration:
    """Integration tests for the embedding system."""
    
    def test_full_workflow_tfidf(self, embedding_system, sample_chunks):
        """Test complete TF-IDF workflow."""
        # Create embeddings
        embeddings = embedding_system.create_embeddings(sample_chunks)
        
        # Save embeddings
        embedding_system.save_embeddings()
        
        # Create new system and load
        new_system = EmbeddingSystem(embedding_system.config)
        loaded = new_system.load_embeddings()
        
        assert loaded is True
        
        # Test query embedding
        query_embedding = new_system.embed_query("machine learning")
        assert query_embedding is not None
        
        # Verify consistency
        assert new_system.chunks == sample_chunks
        np.testing.assert_array_equal(
            new_system.embeddings.toarray(),
            embeddings.toarray()
        )
    
    def test_empty_chunks_handling(self, embedding_system):
        """Test handling of empty chunks list."""
        empty_chunks = []
        embeddings = embedding_system.create_embeddings(empty_chunks)
        
        assert embeddings is not None
        assert embeddings.shape[0] == 0
        assert embedding_system.chunks == empty_chunks
    
    def test_single_chunk_handling(self, embedding_system):
        """Test handling of single chunk."""
        single_chunk = [{
            'text': 'Single test document.',
            'source': 'test.txt',
            'chunk_index': 0,
            'total_chunks': 1
        }]
        
        embeddings = embedding_system.create_embeddings(single_chunk)
        
        assert embeddings is not None
        assert embeddings.shape[0] == 1
        assert embedding_system.chunks == single_chunk
    
    def test_large_text_chunks(self, embedding_system):
        """Test handling of large text chunks."""
        large_chunks = [{
            'text': 'Large text ' * 1000,  # Very long text
            'source': 'large.txt',
            'chunk_index': 0,
            'total_chunks': 1
        }]
        
        embeddings = embedding_system.create_embeddings(large_chunks)
        
        assert embeddings is not None
        assert embeddings.shape[0] == 1

class TestEmbeddingSystemEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_special_characters_in_text(self, embedding_system):
        """Test handling of special characters."""
        special_chunks = [{
            'text': 'Text with émojis 🚀 and spéciał chåractërs!',
            'source': 'special.txt',
            'chunk_index': 0,
            'total_chunks': 1
        }]
        
        embeddings = embedding_system.create_embeddings(special_chunks)
        assert embeddings is not None
        assert embeddings.shape[0] == 1
    
    def test_numeric_only_text(self, embedding_system):
        """Test handling of numeric-only text."""
        numeric_chunks = [{
            'text': '123 456 789 000',
            'source': 'numbers.txt',
            'chunk_index': 0,
            'total_chunks': 1
        }]
        
        embeddings = embedding_system.create_embeddings(numeric_chunks)
        assert embeddings is not None
        assert embeddings.shape[0] == 1
    
    def test_whitespace_only_text(self, embedding_system):
        """Test handling of whitespace-only text."""
        whitespace_chunks = [{
            'text': '   \n\t   ',
            'source': 'whitespace.txt',
            'chunk_index': 0,
            'total_chunks': 1
        }]
        
        embeddings = embedding_system.create_embeddings(whitespace_chunks)
        assert embeddings is not None
        assert embeddings.shape[0] == 1

# Utility functions for testing
def create_mock_openai_response(embedding_dim=1536):
    """Create a mock OpenAI API response."""
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1] * embedding_dim)]
    return mock_response

def assert_embeddings_valid(embeddings, expected_chunks):
    """Assert that embeddings are valid."""
    assert embeddings is not None
    assert embeddings.shape[0] == len(expected_chunks)
    assert embeddings.shape[1] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])