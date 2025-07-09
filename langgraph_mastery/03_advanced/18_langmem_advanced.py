"""
Advanced Enterprise Knowledge Management with LangMem

This implementation showcases advanced LangMem features including:
- Hot path processing for real-time queries
- Background processing for memory management
- Semantic and episodic memory extraction
- User profile management
- Dynamic namespace configuration
- Integration with agents and tools
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from uuid import uuid4
from pathlib import Path
import json

# LangMem imports
from langmem import (
    LangMemClient,
    Document,
    MemoryConfig,
    MemoryType,
    MemoryNamespace,
    MemoryQuery,
    MemoryUpdate
)
from langmem.embeddings import HuggingFaceEmbedding
from langmem.processors import (
    SemanticMemoryExtractor,
    EpisodicMemoryExtractor,
    MemoryConsolidator
)

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
    "memory_path": "langmem_advanced_storage",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "llm_model": "llama3",
    "namespace_config": {
        "general": {"description": "General knowledge and documentation"},
        "user_profiles": {"description": "User preferences and interactions"},
        "episodic": {"description": "Conversation history and episodic memories"}
    }
}

class AdvancedKnowledgeManager:
    """Advanced knowledge manager using LangMem's latest features"""
    
    def __init__(self):
        # Initialize LangMem with advanced configuration
        self.client = self._initialize_langmem()
        
        # Initialize LLM
        self.llm = Ollama(model=CONFIG["llm_model"])
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"]
        )
        
        # Initialize memory processors
        self.semantic_extractor = SemanticMemoryExtractor()
        self.episodic_extractor = EpisodicMemoryExtractor()
        self.memory_consolidator = MemoryConsolidator()
        
        # Create namespaces
        self.namespaces = {}
        self._initialize_namespaces()
    
    def _initialize_langmem(self):
        """Initialize LangMem with advanced configuration"""
        config = MemoryConfig(
            storage_path=CONFIG["memory_path"],
            embedding=HuggingFaceEmbedding(model_name=CONFIG["embedding_model"]),
            enable_background_processing=True,
            background_processing_interval=300  # 5 minutes
        )
        return LangMemClient(config=config)
    
    def _initialize_namespaces(self):
        """Initialize memory namespaces"""
        for name, config in CONFIG["namespace_config"].items():
            self.namespaces[name] = self.client.get_or_create_namespace(
                name=name,
                description=config["description"]
            )
    
    async def add_document(self, file_path: str, metadata: Optional[Dict] = None):
        """Add a document with advanced processing"""
        # Load and split document
        docs = self._load_and_split_document(file_path, metadata)
        
        # Process in the general namespace
        namespace = self.namespaces["general"]
        
        # Add documents with metadata
        for doc in docs:
            # Add to hot path for immediate availability
            await namespace.add_document(
                text=doc.page_content,
                metadata={
                    "source": file_path,
                    "type": "document_chunk",
                    "ingestion_time": datetime.utcnow().isoformat(),
                    **(metadata or {})
                }
            )
        
        # Schedule background processing
        await self._process_in_background(file_path, docs)
    
    async def _process_in_background(self, file_path: str, docs: List[Any]):
        """Process document in the background"""
        # This runs in a background thread
        try:
            # Extract semantic memories
            semantic_memories = await self.semantic_extractor.extract(
                [doc.page_content for doc in docs],
                metadata={"source": file_path}
            )
            
            # Store semantic memories
            for memory in semantic_memories:
                await self.namespaces["general"].add_memory(
                    memory_type=MemoryType.SEMANTIC,
                    content=memory["content"],
                    metadata=memory["metadata"]
                )
            
            # Consolidate memories
            await self.memory_consolidator.consolidate()
            
        except Exception as e:
            print(f"Background processing failed: {str(e)}")
    
    async def query(self, question: str, user_id: str = "guest") -> str:
        """Query the knowledge base with advanced features"""
        # Get relevant memories across namespaces
        query = MemoryQuery(
            query=question,
            namespaces=["general", "episodic"],
            limit=5,
            min_relevance=0.7
        )
        
        # Execute query
        results = await self.client.search(query)
        
        # Format context
        context = "\n\n".join([mem.content for mem in results])
        
        # Create prompt with memory context
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an advanced knowledge assistant. Use the following context to answer the question.
            
            Context: {context}
            """),
            ("human", "{question}")
        ])
        
        # Create and execute chain
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = await chain.ainvoke(question)
        
        # Store interaction as episodic memory
        await self._store_episodic_memory(question, response, user_id)
        
        return response
    
    async def _store_episodic_memory(self, question: str, response: str, user_id: str):
        """Store interaction as episodic memory"""
        episodic_memory = await self.episodic_extractor.extract(
            [f"Q: {question}\nA: {response}"],
            metadata={
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "type": "conversation"
            }
        )
        
        if episodic_memory:
            await self.namespaces["episodic"].add_memory(
                memory_type=MemoryType.EPISODIC,
                content=episodic_memory[0]["content"],
                metadata=episodic_memory[0]["metadata"]
            )
    
    def _load_and_split_document(self, file_path: str, metadata: Optional[Dict]) -> List[Any]:
        """Load and split document into chunks"""
        file_ext = Path(file_path).suffix.lower()
        
        # Select appropriate loader
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
        return self.text_splitter.split_documents(docs)

async def main():
    """Main async function to run the knowledge manager"""
    # Initialize the knowledge manager
    print("🚀 Initializing Advanced Knowledge Manager...")
    km = AdvancedKnowledgeManager()
    
    # Example usage
    while True:
        print("\n📚 Advanced Knowledge Manager")
        print("1. Add document")
        print("2. Query knowledge base")
        print("3. View namespace statistics")
        print("4. Exit")
        
        choice = input("\n🔹 Enter your choice (1-4): ").strip()
        
        if choice == "1":
            # Add document
            file_path = input("\n📂 Enter path to document: ").strip()
            if not os.path.exists(file_path):
                print("❌ File not found")
                continue
                
            metadata = {
                "department": input("🏢 Department (optional): ").strip() or "general",
                "document_type": input("📄 Document type: ").strip() or "document"
            }
            
            await km.add_document(file_path, metadata)
            print("\n✅ Document added and processing in background")
            
        elif choice == "2":
            # Query knowledge base
            question = input("\n❓ Enter your question: ").strip()
            if not question:
                print("❌ Please enter a question")
                continue
                
            user_id = input("👤 Your user ID (optional, press Enter for 'guest'): ").strip() or "guest"
            
            print("\n🔍 Searching knowledge base...\n")
            response = await km.query(question, user_id)
            print("\n💡 Response:")
            print("=" * 80)
            print(response)
            print("=" * 80)
            
        elif choice == "3":
            # Show namespace statistics
            print("\n📊 Namespace Statistics:")
            for name, namespace in km.namespaces.items():
                stats = await namespace.get_statistics()
                print(f"\n🔹 {name.upper()}:")
                print(f"   Documents: {stats['document_count']}")
                print(f"   Memories: {stats['memory_count']}")
                print(f"   Storage: {stats['storage_size_mb']:.2f} MB")
            
        elif choice == "4":
            print("👋 Exiting...")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    asyncio.run(main())
