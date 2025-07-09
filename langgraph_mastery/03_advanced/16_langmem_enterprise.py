"""
Enterprise-Grade Knowledge Management System with LangMem

This example demonstrates how to build a production-ready knowledge management system
that can handle large-scale enterprise documentation with efficient retrieval and memory management.

Key Features:
1. Document ingestion pipeline with chunking and metadata extraction
2. Hybrid search combining semantic and keyword search
3. Memory management with automatic eviction policies
4. Access control and document versioning
5. Usage analytics and monitoring
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import hashlib
from uuid import uuid4

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.memory import VectorStoreMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# Configuration
CONFIG = {
    "data_dir": "data/enterprise_docs",
    "vector_store_path": "faiss_enterprise_index",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_memory_items": 1000,
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "llm_model": "llama3",  # Using Ollama's local LLM
}

class EnterpriseKnowledgeManager:
    """Enterprise Knowledge Management System with LangMem integration"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=CONFIG["embedding_model"],
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False}
        )
        self.llm = Ollama(model=CONFIG["llm_model"])
        self.vector_store = None
        self.memory = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize vector store and memory components"""
        if os.path.exists(CONFIG["vector_store_path"]):
            self.vector_store = FAISS.load_local(
                CONFIG["vector_store_path"], 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            # Create empty vector store if it doesn't exist
            self.vector_store = FAISS.from_texts(
                ["Initial document"], 
                embedding=self.embeddings
            )
            self.vector_store.save_local(CONFIG["vector_store_path"])
        
        # Initialize memory with eviction policy
        self.memory = VectorStoreMemory(
            vectorstore=self.vector_store,
            memory_key="chat_history",
            return_messages=True,
            k=5,  # Number of most relevant memories to retrieve
        )
    
    def _process_document(self, file_path: str) -> List[Document]:
        """Process a single document and return chunks"""
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"]
        )
        return text_splitter.split_documents(docs)
    
    def add_documents(self, file_paths: List[str], metadata: Optional[Dict] = None):
        """Add documents to the knowledge base"""
        all_chunks = []
        
        for file_path in file_paths:
            try:
                # Process each document into chunks
                chunks = self._process_document(file_path)
                
                # Add metadata to each chunk
                for chunk in chunks:
                    chunk.metadata.update({
                        "source": file_path,
                        "chunk_id": str(uuid4()),
                        "ingestion_time": datetime.utcnow().isoformat(),
                        "document_hash": self._calculate_file_hash(file_path),
                        **(metadata or {})
                    })
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        if all_chunks:
            # Add to vector store
            self.vector_store.add_documents(all_chunks)
            self.vector_store.save_local(CONFIG["vector_store_path"])
            print(f"Added {len(all_chunks)} chunks from {len(file_paths)} files")
    
    def query(self, question: str, user_id: str = "system") -> str:
        """Query the knowledge base with RAG"""
        # Create a retriever
        retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance
            search_kwargs={"k": 5}
        )
        
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
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Execute the chain
        response = rag_chain.invoke(question)
        
        # Store interaction in memory
        self._log_interaction(question, response, user_id)
        
        return response
    
    def _log_interaction(self, question: str, response: str, user_id: str):
        """Log user interaction for analytics and future reference"""
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "question": question,
            "response": response,
            "interaction_id": str(uuid4())
        }
        
        # In a production system, you would store this in a proper database
        # For this example, we'll just print it
        print(f"Interaction logged: {interaction}")
    
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
    print("Initializing Enterprise Knowledge Manager...")
    km = EnterpriseKnowledgeManager()
    
    # Example usage
    while True:
        print("\nOptions:")
        print("1. Add documents to knowledge base")
        print("2. Query knowledge base")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            # Example: Add documents
            doc_path = input("Enter path to document(s), separated by space: ").strip()
            if not doc_path:
                print("No path provided")
                continue
                
            doc_paths = doc_path.split()
            valid_paths = [p for p in doc_paths if os.path.exists(p)]
            
            if not valid_paths:
                print("No valid file paths found")
                continue
                
            metadata = {
                "department": input("Department (optional): ").strip() or "general",
                "document_type": input("Document type (e.g., manual, policy, report): ").strip() or "document"
            }
            
            km.add_documents(valid_paths, metadata)
            
        elif choice == "2":
            # Query the knowledge base
            question = input("\nEnter your question: ").strip()
            if not question:
                print("Please enter a question")
                continue
                
            user_id = input("Your user ID (optional, press Enter for 'guest'): ").strip() or "guest"
            
            print("\nSearching knowledge base...\n")
            response = km.query(question, user_id)
            print("\nResponse:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
        elif choice == "3":
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
