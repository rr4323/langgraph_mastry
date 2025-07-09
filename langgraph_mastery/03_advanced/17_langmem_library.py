"""
Enterprise Knowledge Management System using LangMem

This version demonstrates how to implement the same functionality as 16_langmem_enterprise.py
but using the LangMem library for memory management.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import hashlib
from uuid import uuid4

# LangMem specific imports
from langmem import LangMemClient, Document
from langmem.embeddings import HuggingFaceEmbedding
from langmem.retrievers import VectorRetriever

# Other imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration
CONFIG = {
    "data_dir": "data/enterprise_docs",
    "memory_path": "langmem_storage",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "llm_model": "llama3",
}

class LangMemKnowledgeManager:
    """Knowledge Manager using LangMem library"""
    
    def __init__(self):
        # Initialize LangMem client
        self.client = LangMemClient(
            embedding=HuggingFaceEmbedding(model_name=CONFIG["embedding_model"]),
            storage_path=CONFIG["memory_path"]
        )
        
        # Initialize LLM
        self.llm = Ollama(model=CONFIG["llm_model"])
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"]
        )
        
        # Create or load memory collection
        self.collection = self.client.get_or_create_collection("enterprise_knowledge")
    
    def _process_document(self, file_path: str) -> List[Document]:
        """Process a single document and return LangMem Document objects"""
        file_ext = Path(file_path).suffix.lower()
        
        # Select appropriate loader based on file type
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif file_ext in ['.txt', '.md']:
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Load and split document
        docs = loader.load()
        all_chunks = []
        
        for doc in docs:
            # Split into chunks
            chunks = self.text_splitter.split_text(doc.page_content)
            
            # Create LangMem Document objects
            for i, chunk in enumerate(chunks):
                doc_id = f"{Path(file_path).stem}_chunk_{i}"
                metadata = {
                    "source": file_path,
                    "chunk_id": str(uuid4()),
                    "ingestion_time": datetime.utcnow().isoformat(),
                    "document_hash": self._calculate_file_hash(file_path),
                    **doc.metadata
                }
                
                all_chunks.append(Document(
                    id=doc_id,
                    text=chunk,
                    metadata=metadata
                ))
        
        return all_chunks
    
    def add_documents(self, file_paths: List[str], metadata: Optional[Dict] = None):
        """Add documents to the knowledge base"""
        all_docs = []
        
        for file_path in file_paths:
            try:
                print(f"Processing {file_path}...")
                docs = self._process_document(file_path)
                all_docs.extend(docs)
                print(f"  → Extracted {len(docs)} chunks")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        if all_docs:
            # Add to LangMem collection
            self.collection.add_documents(all_docs)
            print(f"\n✅ Successfully added {len(all_docs)} chunks to knowledge base")
    
    def query(self, question: str, user_id: str = "system") -> str:
        """Query the knowledge base using RAG"""
        # Retrieve relevant documents using LangMem's retriever
        retriever = VectorRetriever(collection=self.collection, top_k=5)
        
        # Get relevant chunks
        results = retriever.retrieve(question)
        
        # Format context for the prompt
        context = "\n\n".join([doc.text for doc in results])
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an enterprise knowledge assistant. Use the following context to answer the question. 
            If you don't know the answer, say you don't know. Keep the answer concise and professional.
            
            Context: {context}
            """),
            ("human", "{question}")
        ])
        
        # Create RAG chain
        rag_chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Execute the chain
        response = rag_chain.invoke(question)
        
        # Log interaction
        self._log_interaction(question, response, user_id)
        
        return response
    
    def _log_interaction(self, question: str, response: str, user_id: str):
        """Log user interaction"""
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "question": question,
            "response": response,
            "interaction_id": str(uuid4())
        }
        
        # In a real application, you would store this in a proper database
        print(f"\n🔍 Interaction logged for user {user_id}:")
        print(f"   Question: {question}")
        print(f"   Response: {response[:200]}..." if len(response) > 200 else f"   Response: {response}")
    
    @staticmethod
    def _calculate_file_hash(file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

def main():
    # Initialize the knowledge manager
    print("🚀 Initializing LangMem Knowledge Manager...")
    km = LangMemKnowledgeManager()
    
    # Example usage
    while True:
        print("\n📚 Options:")
        print("1. Add documents to knowledge base")
        print("2. Query knowledge base")
        print("3. View collection stats")
        print("4. Exit")
        
        choice = input("\n🔹 Enter your choice (1-4): ").strip()
        
        if choice == "1":
            # Add documents
            doc_path = input("\n📂 Enter path to document(s), separated by space: ").strip()
            if not doc_path:
                print("❌ No path provided")
                continue
                
            doc_paths = doc_path.split()
            valid_paths = [p for p in doc_paths if os.path.exists(p)]
            
            if not valid_paths:
                print("❌ No valid file paths found")
                continue
                
            metadata = {
                "department": input("🏢 Department (optional): ").strip() or "general",
                "document_type": input("📄 Document type (e.g., manual, policy, report): ").strip() or "document"
            }
            
            km.add_documents(valid_paths, metadata)
            
        elif choice == "2":
            # Query the knowledge base
            question = input("\n❓ Enter your question: ").strip()
            if not question:
                print("❌ Please enter a question")
                continue
                
            user_id = input("👤 Your user ID (optional, press Enter for 'guest'): ").strip() or "guest"
            
            print("\n🔍 Searching knowledge base...\n")
            response = km.query(question, user_id)
            print("\n💡 Response:")
            print("=" * 80)
            print(response)
            print("=" * 80)
            
        elif choice == "3":
            # Show collection stats
            stats = km.collection.get_stats()
            print("\n📊 Collection Statistics:")
            print(f"   Documents: {stats['document_count']}")
            print(f"   Total Chunks: {stats['chunk_count']}")
            print(f"   Storage Size: {stats['storage_size_mb']:.2f} MB")
            
        elif choice == "4":
            print("👋 Exiting...")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
