# ============================================================================
# FILE: src/qa_system.py
# ============================================================================
from typing import List, Dict, Optional
from src.document_processor import DocumentProcessor
from src.embedding_system import EmbeddingSystem
from src.retriever import Retriever
from config import Config

# Optional OpenAI import for completion
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class RAGSystem:
    """Main RAG question-answering system."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.document_processor = DocumentProcessor(self.config)
        self.embedding_system = EmbeddingSystem(self.config)
        self.retriever = None
        self.is_initialized = False
        
        self.use_openai_completion = (
            OPENAI_AVAILABLE and 
            self.config.OPENAI_API_KEY and
            self.config.USE_OPENAI_EMBEDDINGS
        )
    
    def initialize(self, force_rebuild: bool = False):
        """Initialize the RAG system by loading or creating embeddings."""
        print("Initializing RAG system...")
        
        # Try to load existing embeddings
        if not force_rebuild and self.embedding_system.load_embeddings():
            print("Loaded existing embeddings")
        else:
            print("Creating new embeddings...")
            # Load documents
            documents = self.document_processor.load_documents()
            
            if not documents:
                print("No documents found! Please add .txt files to the documents/ folder")
                return False
            
            # Process documents into chunks
            chunks = self.document_processor.process_documents(documents)
            
            if not chunks:
                print("No chunks created from documents")
                return False
            
            # Create embeddings
            self.embedding_system.create_embeddings(chunks)
            self.embedding_system.save_embeddings()
        
        # Initialize retriever
        self.retriever = Retriever(self.embedding_system, self.config)
        self.is_initialized = True
        
        print("RAG system initialized successfully!")
        return True
    
    def ask(self, question: str) -> Dict:
        """Ask a question and get an answer with context."""
        if not self.is_initialized:
            return {
                'answer': "System not initialized. Please run initialize() first.",
                'context': "",
                'chunks': []
            }
        
        # Retrieve relevant chunks
        relevant_chunks = self.retriever.retrieve_relevant_chunks(question)
        
        if not relevant_chunks:
            return {
                'answer': "I don't have enough information to answer that question.",
                'context': "",
                'chunks': []
            }
        
        # Format context
        context = self.retriever.format_context(relevant_chunks)
        
        # Generate answer
        if self.use_openai_completion:
            answer = self._generate_openai_answer(question, context)
        else:
            answer = self._generate_simple_answer(question, relevant_chunks)
        
        return {
            'answer': answer,
            'context': context,
            'chunks': relevant_chunks
        }
    
    def _generate_openai_answer(self, question: str, context: str) -> str:
        """Generate answer using OpenAI completion."""
        prompt = f"""Context from documents:
{context}

Question: {question}

Please answer based on the provided context. If the answer isn't in the context, say "I don't have enough information to answer that question."

Answer:"""
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating OpenAI response: {e}")
            return self._generate_simple_answer(question, [])
    
    def _generate_simple_answer(self, question: str, chunks: List[Dict]) -> str:
        """Generate a simple answer by returning the most relevant chunk."""
        if not chunks:
            return "I don't have enough information to answer that question."
        
        best_chunk = chunks[0]
        return f"Based on the documentation: {best_chunk['text'][:300]}..."