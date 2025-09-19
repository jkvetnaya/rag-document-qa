# ============================================================================
# FILE: tests/test_document_processor.py
# ============================================================================
import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, mock_open
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.document_processor import DocumentProcessor
from config import Config

class TestConfig(Config):
    """Test configuration with temporary paths."""
    def __init__(self, temp_dir):
        super().__init__()
        self.DOCUMENTS_PATH = os.path.join(temp_dir, "documents")
        self.DATA_PATH = os.path.join(temp_dir, "data")
        self.CHUNK_SIZE = 100  # Smaller for testing
        self.CHUNK_OVERLAP = 20

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    return TestConfig(temp_dir)

@pytest.fixture
def document_processor(test_config):
    """Create a document processor instance."""
    return DocumentProcessor(test_config)

@pytest.fixture
def sample_documents_dir(test_config):
    """Create a directory with sample documents."""
    docs_dir = test_config.DOCUMENTS_PATH
    os.makedirs(docs_dir, exist_ok=True)
    
    # Create sample files
    test_files = {
        "doc1.txt": "This is the first document. It contains some text about machine learning and AI.",
        "doc2.txt": "The second document discusses natural language processing and neural networks.",
        "doc3.txt": "A third document with information about data science and analytics.",
        "not_txt.pdf": "This should be ignored",
        "empty.txt": "",
        "special_chars.txt": "Document with special characters: émojis 🚀 and spéciał chåractërs!"
    }
    
    for filename, content in test_files.items():
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return docs_dir

class TestDocumentProcessorInitialization:
    """Test document processor initialization."""
    
    def test_init_with_config(self, test_config):
        """Test initialization with custom config."""
        processor = DocumentProcessor(test_config)
        assert processor.config == test_config
    
    def test_init_without_config(self):
        """Test initialization with default config."""
        processor = DocumentProcessor()
        assert processor.config is not None
        assert isinstance(processor.config, Config)
    
    def test_config_values_used(self, test_config):
        """Test that config values are properly used."""
        processor = DocumentProcessor(test_config)
        assert processor.config.CHUNK_SIZE == 100
        assert processor.config.CHUNK_OVERLAP == 20

class TestLoadDocuments:
    """Test document loading functionality."""
    
    def test_load_documents_existing_directory(self, document_processor, sample_documents_dir):
        """Test loading documents from existing directory."""
        documents = document_processor.load_documents()
        
        # Should load only .txt files
        expected_files = {"doc1.txt", "doc2.txt", "doc3.txt", "empty.txt", "special_chars.txt"}
        assert set(documents.keys()) == expected_files
        
        # Check content
        assert "machine learning" in documents["doc1.txt"]
        assert "natural language processing" in documents["doc2.txt"]
        assert documents["empty.txt"] == ""
        assert "émojis 🚀" in documents["special_chars.txt"]
    
    def test_load_documents_custom_path(self, document_processor, temp_dir):
        """Test loading documents from custom path."""
        custom_dir = os.path.join(temp_dir, "custom_docs")
        os.makedirs(custom_dir)
        
        # Create test file
        test_file = os.path.join(custom_dir, "custom.txt")
        with open(test_file, 'w') as f:
            f.write("Custom document content")
        
        documents = document_processor.load_documents(custom_dir)
        
        assert "custom.txt" in documents
        assert documents["custom.txt"] == "Custom document content"
    
    def test_load_documents_nonexistent_directory(self, document_processor):
        """Test loading documents from non-existent directory."""
        documents = document_processor.load_documents("nonexistent_path")
        
        # Should create directory and return empty dict
        assert documents == {}
        assert os.path.exists("nonexistent_path")
        
        # Cleanup
        shutil.rmtree("nonexistent_path")
    
    def test_load_documents_empty_directory(self, document_processor, temp_dir):
        """Test loading documents from empty directory."""
        empty_dir = os.path.join(temp_dir, "empty")
        os.makedirs(empty_dir)
        
        documents = document_processor.load_documents(empty_dir)
        assert documents == {}
    
    def test_load_documents_no_txt_files(self, document_processor, temp_dir):
        """Test loading documents from directory with no .txt files."""
        no_txt_dir = os.path.join(temp_dir, "no_txt")
        os.makedirs(no_txt_dir)
        
        # Create non-.txt files
        with open(os.path.join(no_txt_dir, "doc.pdf"), 'w') as f:
            f.write("PDF content")
        with open(os.path.join(no_txt_dir, "doc.docx"), 'w') as f:
            f.write("DOCX content")
        
        documents = document_processor.load_documents(no_txt_dir)
        assert documents == {}
    
'''
    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_load_documents_permission_error(self, mock_file, document_processor, temp_dir):
        """Test handling of permission errors when loading documents."""
        # Create directory with a file
        test_dir = os.path.join(temp_dir, "permission_test")
        os.makedirs(test_dir)
        
        # Create a file that will trigger permission error when opened
        test_file = os.path.join(test_dir, "restricted.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        # Mock should only affect the read operation, not file creation
        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = PermissionError("Permission denied")
            
            documents = document_processor.load_documents(test_dir)
            assert documents == {}
'''

class TestChunkText:
    """Test text chunking functionality."""
    
    def test_chunk_text_basic(self, document_processor):
        """Test basic text chunking."""
        text = "This is a test document. " * 10  # 250 characters
        chunks = document_processor.chunk_text(text, chunk_size=50, overlap=10)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 60 for chunk in chunks)  # Allow some flexibility
        
        # Check overlap
        if len(chunks) > 1:
            # Should have some overlap between consecutive chunks
            assert chunks[0][-5:] in chunks[1] or chunks[1][:5] in chunks[0]
    
    def test_chunk_text_custom_parameters(self, document_processor):
        """Test chunking with custom parameters."""
        text = "Word " * 50  # 250 characters
        chunk_size = 30
        overlap = 5
        
        chunks = document_processor.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= chunk_size + 10 for chunk in chunks)  # Allow boundary flexibility
    
    def test_chunk_text_default_parameters(self, document_processor):
        """Test chunking with default parameters from config."""
        text = "Default parameters test. " * 20  # 500 characters
        chunks = document_processor.chunk_text(text)
        
        # Should use config values
        expected_chunk_size = document_processor.config.CHUNK_SIZE
        assert len(chunks) >= 1
    
    def test_chunk_text_short_text(self, document_processor):
        """Test chunking of text shorter than chunk size."""
        text = "Short text."
        chunks = document_processor.chunk_text(text, chunk_size=100, overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_sentence_boundaries(self, document_processor):
        """Test that chunking respects sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = document_processor.chunk_text(text, chunk_size=30, overlap=5)
        
        # Most chunks should end with a period (sentence boundary)
        sentence_endings = sum(1 for chunk in chunks if chunk.rstrip().endswith('.'))
        assert sentence_endings >= len(chunks) // 2  # At least half should respect boundaries
    
    def test_chunk_text_whitespace_handling(self, document_processor):
        """Test proper whitespace handling in chunks."""
        text = "Text   with    multiple     spaces\n\nand\t\ttabs."
        chunks = document_processor.chunk_text(text, chunk_size=20, overlap=5)
        
        # Should normalize whitespace
        for chunk in chunks:
            assert "  " not in chunk  # No double spaces
            assert "\n" not in chunk  # No newlines
            assert "\t" not in chunk  # No tabs
    
    def test_chunk_text_empty_string(self, document_processor):
        """Test chunking of empty string."""
        chunks = document_processor.chunk_text("", chunk_size=100, overlap=10)
        assert chunks == []
    
    def test_chunk_text_whitespace_only(self, document_processor):
        """Test chunking of whitespace-only string."""
        chunks = document_processor.chunk_text("   \n\t   ", chunk_size=100, overlap=10)
        assert chunks == []
    
'''
    def test_chunk_text_special_characters(self, document_processor):
        """Test chunking text with special characters."""
        text = "Text with émojis 🚀 and spéciał chåractërs! More text follows."
        chunks = document_processor.chunk_text(text, chunk_size=30, overlap=5)
        
        assert len(chunks) >= 1
        # Special characters should be preserved
        combined_text = " ".join(chunks)
        assert "émojis 🚀" in combined_text
        assert "spéciał chåractërs" in combined_text
'''
    
'''
    def test_chunk_text_very_long_word(self, document_processor):
        """Test chunking with very long words."""
        long_word = "a" * 150  # Word longer than typical chunk size
        text = f"Start {long_word} end."
        
        chunks = document_processor.chunk_text(text, chunk_size=100, overlap=10)
        
        assert len(chunks) >= 1
        # Long word should be preserved in at least one chunk
        assert any(long_word in chunk for chunk in chunks)
'''
class TestProcessDocuments:
    """Test document processing functionality."""
    
    def test_process_documents_basic(self, document_processor):
        """Test basic document processing."""
        documents = {
            "doc1.txt": "First document content. This has multiple sentences.",
            "doc2.txt": "Second document with different content."
        }
        
        chunks = document_processor.process_documents(documents)
        
        assert len(chunks) > 0
        
        # Check chunk structure
        for chunk in chunks:
            assert 'text' in chunk
            assert 'source' in chunk
            assert 'chunk_index' in chunk
            assert 'total_chunks' in chunk
            assert chunk['source'] in documents.keys()
            assert isinstance(chunk['chunk_index'], int)
            assert isinstance(chunk['total_chunks'], int)
            assert chunk['chunk_index'] < chunk['total_chunks']
    
    def test_process_documents_metadata(self, document_processor):
        """Test that document processing creates correct metadata."""
        documents = {
            "single.txt": "Short document.",
            "multi.txt": "Much longer document that will be split into multiple chunks. " * 10
        }
        
        chunks = document_processor.process_documents(documents)
        
        # Group chunks by source
        chunks_by_source = {}
        for chunk in chunks:
            source = chunk['source']
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)
        
        # Check chunk indices and totals
        for source, source_chunks in chunks_by_source.items():
            total_chunks = source_chunks[0]['total_chunks']
            assert len(source_chunks) == total_chunks
            
            # Check indices are consecutive and start from 0
            indices = [chunk['chunk_index'] for chunk in source_chunks]
            assert indices == list(range(total_chunks))
    
    def test_process_documents_empty_dict(self, document_processor):
        """Test processing empty documents dictionary."""
        chunks = document_processor.process_documents({})
        assert chunks == []
    
    def test_process_documents_empty_content(self, document_processor):
        """Test processing documents with empty content."""
        documents = {
            "empty1.txt": "",
            "empty2.txt": "   ",
            "normal.txt": "Normal content here."
        }
        
        chunks = document_processor.process_documents(documents)
        
        # Should only create chunks for non-empty content
        sources = [chunk['source'] for chunk in chunks]
        assert "normal.txt" in sources
        # Empty documents might or might not create chunks depending on implementation
    
    def test_process_documents_single_document(self, document_processor):
        """Test processing a single document."""
        documents = {"single.txt": "Single document content for testing purposes."}
        
        chunks = document_processor.process_documents(documents)
        
        assert len(chunks) >= 1
        assert all(chunk['source'] == "single.txt" for chunk in chunks)
    
    def test_process_documents_large_content(self, document_processor):
        """Test processing documents with large content."""
        large_content = "Large document content. " * 100  # 2400 characters
        documents = {"large.txt": large_content}
        
        chunks = document_processor.process_documents(documents)
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        
        # Verify all chunks belong to the same document
        assert all(chunk['source'] == "large.txt" for chunk in chunks)
        
        # Verify chunk indices
        total_chunks = chunks[0]['total_chunks']
        assert len(chunks) == total_chunks
        
        indices = [chunk['chunk_index'] for chunk in chunks]
        assert sorted(indices) == list(range(total_chunks))

class TestDocumentProcessorIntegration:
    """Integration tests for document processor."""
    
    def test_full_workflow(self, document_processor, sample_documents_dir):
        """Test complete workflow: load → process."""
        # Load documents
        documents = document_processor.load_documents()
        assert len(documents) > 0
        
        # Process documents
        chunks = document_processor.process_documents(documents)
        assert len(chunks) > 0
        
        # Verify all source documents are represented
        sources_in_chunks = set(chunk['source'] for chunk in chunks)
        expected_sources = set(name for name in documents.keys() if documents[name].strip())
        
        # All non-empty documents should have chunks
        assert expected_sources.issubset(sources_in_chunks)
    
    def test_workflow_with_custom_config(self, temp_dir):
        """Test workflow with custom configuration."""
        # Create custom config
        config = TestConfig(temp_dir)
        config.CHUNK_SIZE = 50
        config.CHUNK_OVERLAP = 10
        
        processor = DocumentProcessor(config)
        
        # Create test documents
        docs_dir = config.DOCUMENTS_PATH
        os.makedirs(docs_dir)
        
        test_content = "Custom configuration test. " * 10
        with open(os.path.join(docs_dir, "test.txt"), 'w') as f:
            f.write(test_content)
        
        # Process with custom settings
        documents = processor.load_documents()
        chunks = processor.process_documents(documents)
        
        assert len(chunks) > 0
        # Chunks should respect custom size (approximately)
        assert all(len(chunk['text']) <= config.CHUNK_SIZE + 20 for chunk in chunks)

class TestDocumentProcessorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_unicode_handling(self, document_processor):
        """Test proper Unicode handling."""
        unicode_documents = {
            "unicode.txt": "Document with Unicode: 你好世界 καλημέρα العالم مرحبا"
        }
        
        chunks = document_processor.process_documents(unicode_documents)
        
        assert len(chunks) > 0
        combined_text = " ".join(chunk['text'] for chunk in chunks)
        assert "你好世界" in combined_text
        assert "καλημέρα" in combined_text
        assert "العالم" in combined_text
    
    def test_very_short_chunks_config(self, temp_dir):
        """Test with very short chunk configuration."""
        config = TestConfig(temp_dir)
        config.CHUNK_SIZE = 10  # Very small chunks
        config.CHUNK_OVERLAP = 2
        
        processor = DocumentProcessor(config)
        
        text = "This is a test document with multiple words."
        chunks = processor.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 15 for chunk in chunks)  # Allow some flexibility
    
'''
    def test_zero_overlap_config(self, temp_dir):
        """Test with zero overlap configuration."""
        config = TestConfig(temp_dir)
        config.CHUNK_OVERLAP = 0
        
        processor = DocumentProcessor(config)
        
        text = "First sentence. Second sentence. Third sentence."
        chunks = processor.chunk_text(text, chunk_size=20)
        
        assert len(chunks) > 1
        # With zero overlap, chunks should not share content
        if len(chunks) > 1:
            assert chunks[0][-5:] not in chunks[1]

'''
'''
    def test_overlap_larger_than_chunk_size(self, temp_dir):
        """Test handling of overlap larger than chunk size."""
        config = TestConfig(temp_dir)
        config.CHUNK_SIZE = 20
        config.CHUNK_OVERLAP = 30  # Larger than chunk size
        
        processor = DocumentProcessor(config)
        
        text = "Test document with overlapping configuration issue."
        chunks = processor.chunk_text(text)
        
        # Should handle gracefully without infinite loops
        assert len(chunks) > 0
        assert len(chunks) < 100  # Sanity check for no infinite loops

'''

# Utility functions for testing
def create_test_file(directory, filename, content):
    """Create a test file with given content."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath

def count_words_in_chunks(chunks):
    """Count total words across all chunks."""
    total_words = 0
    for chunk in chunks:
        total_words += len(chunk['text'].split())
    return total_words

if __name__ == "__main__":
    pytest.main([__file__, "-v"])