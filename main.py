# ============================================================================
# FILE: main.py
# ============================================================================
import sys
import os
from src.qa_system import RAGSystem
from config import Config


def create_sample_documents():
    """Create sample documents for testing."""
    documents_path = "documents"
    os.makedirs(documents_path, exist_ok=True)
    
    sample_docs = {
        "rag_overview.txt": """
Retrieval Augmented Generation (RAG) is a technique that combines the power of 
large language models with external knowledge sources. RAG systems work by first 
retrieving relevant information from a knowledge base, then using that information 
to generate more accurate and contextual responses.

The key components of a RAG system include document ingestion, text chunking, 
embedding creation, vector storage, similarity search, and answer generation. 
This approach helps reduce hallucinations and provides more factual responses.
        """,
        
        "embedding_systems.txt": """
Embedding systems convert text into numerical vectors that capture semantic meaning. 
There are different approaches to creating embeddings:

1. TF-IDF (Term Frequency-Inverse Document Frequency): A traditional approach that 
   represents documents based on word frequency and rarity.

2. Dense embeddings: Modern approaches like OpenAI's text-embedding models create 
   dense vector representations that capture deeper semantic relationships.

The choice of embedding system affects the quality of retrieval and the overall 
performance of the RAG system.
        """,
        
        "vector_search.txt": """
Vector search, also known as semantic search, finds similar documents by comparing 
their vector representations using metrics like cosine similarity. Unlike traditional 
keyword search, vector search can find conceptually similar content even when exact 
words don't match.

Key concepts include similarity thresholds, top-k retrieval, and ranking algorithms. 
The quality of embeddings directly impacts search effectiveness.
        """
    }
    
    for filename, content in sample_docs.items():
        filepath = os.path.join(documents_path, filename)
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            print(f"Created sample document: {filename}")


def main():
    """Main CLI interface for the RAG system."""
    print("=== RAG Document Question-Answering System ===\n")
    
    # Create sample documents if none exist
    if not os.path.exists("documents") or not os.listdir("documents"):
        print("Creating sample documents...")
        create_sample_documents()
        print()
    
    # Initialize the RAG system
    config = Config()
    rag_system = RAGSystem(config)
    
    if not rag_system.initialize():
        print("Failed to initialize RAG system. Exiting.")
        sys.exit(1)
    
    print("\nRAG system ready! Ask questions about your documents.")
    print("Type 'quit', 'exit', or 'q' to exit.\n")
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print("\nSearching for relevant information...")
            result = rag_system.ask(question)
            
            print(f"\nAnswer: {result['answer']}")
            
            if result['chunks']:
                print(f"\nBased on {len(result['chunks'])} relevant chunks:")
                for i, chunk in enumerate(result['chunks'], 1):
                    score = chunk.get('similarity_score', 0)
                    source = chunk['source']
                    print(f"  {i}. {source} (relevance: {score:.3f})")
            
            print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
