# ============================================================================
# FILE: README.md
# ============================================================================
# RAG Document Question-Answering System

A simple Retrieval Augmented Generation (RAG) system that demonstrates how to build a document question-answering system using vector embeddings and similarity search.

## Features

- Document ingestion and chunking
- TF-IDF or OpenAI embeddings
- Semantic similarity search
- Context-augmented answer generation
- Simple command-line interface

## Setup Instructions

1. **Create the project structure**:
   ```bash
   mkdir rag-document-qa
   cd rag-document-qa
   ```

2. **Set up Python virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add documents**:
   - Place your .txt documents in the `documents/` folder
   - Or run the system and it will create sample documents

5. **Configure (optional)**:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key if you want to use OpenAI embeddings
   - Set `USE_OPENAI_EMBEDDINGS = True` in `config.py`

## Usage

1. **Run the system**:
   ```bash
   python main.py
   ```

2. **Ask questions**:
   - The system will process your documents and create embeddings
   - Ask questions about the content in your documents
   - Type 'quit' to exit

## Configuration

Edit `config.py` to customize:
- Chunk size and overlap
- Number of retrieved chunks (top_k)
- Embedding model settings
- File paths

## Project Structure

```
rag-document-qa/
├── .vscode/              # VS Code configuration
├── src/                  # Main source code
│   ├── document_processor.py
│   ├── embedding_system.py
│   ├── retriever.py
│   └── qa_system.py
├── documents/            # Place your .txt files here
├── data/                 # Stored embeddings and indices
├── tests/                # Unit tests
├── main.py               # Main entry point
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
└── README.md
```

## How It Works

1. **Document Processing**: Load .txt files and split into overlapping chunks
2. **Embedding Creation**: Convert text chunks to vector representations
3. **Query Processing**: Convert questions to the same vector space
4. **Similarity Search**: Find most relevant chunks using cosine similarity
5. **Answer Generation**: Use retrieved context to generate answers

## Example Questions

Try asking questions like:
- "What is RAG?"
- "How do embeddings work?"
- "What's the difference between TF-IDF and dense embeddings?"

## Development

- Use VS Code with the recommended extensions
- Run tests with: `pytest tests/`
- Format code with: `black src/`
- Lint code with: `pylint src/`
