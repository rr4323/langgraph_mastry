# Enterprise Knowledge Management with LangMem

This example demonstrates how to build a production-ready knowledge management system using LangChain and FAISS for vector storage, with features suitable for enterprise deployment.

## Features

- **Document Processing**: Supports multiple file formats (PDF, DOCX, TXT, MD)
- **Efficient Retrieval**: Hybrid search combining semantic and keyword search
- **Memory Management**: Automatic document chunking and vector storage
- **Metadata Support**: Track document sources, versions, and access patterns
- **User Interaction Logging**: Monitor queries and system usage
- **Scalable Architecture**: Designed to handle large document collections

## Prerequisites

- Python 3.8+
- Ollama (for local LLM)
- Required Python packages (install via `pip install -r requirements.txt`)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Ollama service and pull the required model:
   ```bash
   ollama pull llama3
   ```

## Usage

Run the knowledge management system:

```bash
python 16_langmem_enterprise.py
```

### Adding Documents

1. Select option 1 to add documents
2. Enter the path to your document(s)
3. Add optional metadata (department, document type)

### Querying the Knowledge Base

1. Select option 2 to query
2. Enter your question
3. (Optional) Provide a user ID for tracking

## Enterprise Integration Points

This implementation can be extended with:

1. **Authentication & Authorization**: Integrate with enterprise SSO
2. **Scalable Storage**: Replace FAISS with Redis or Pinecone for production
3. **Monitoring**: Add Prometheus metrics and logging
4. **API Layer**: Expose as a REST or gRPC service
5. **Document Preprocessing**: Add OCR for scanned documents

## Performance Considerations

- **Chunk Size**: Adjust `chunk_size` and `chunk_overlap` based on document types
- **Embedding Model**: Choose an appropriate model for your use case
- **Vector Store**: For production, consider a managed vector database
- **Caching**: Implement caching for frequent queries

## Security Considerations

- Store API keys and sensitive data in environment variables
- Implement proper access controls
- Regularly update dependencies
- Monitor for anomalous queries

## Advanced Querying with LangMem

The advanced implementation includes sophisticated querying capabilities:

### Memory Query Features

1. **Cross-Namespace Search**
   ```python
   query = MemoryQuery(
       query=question,
       namespaces=["general", "episodic"],  # Search across multiple namespaces
       limit=5,                            # Number of results to return
       min_relevance=0.7                   # Minimum relevance score (0-1)
   )
   ```

2. **Context-Aware Responses**
   - Combines relevant memories from different namespaces
   - Uses semantic similarity to find the most relevant context
   - Automatically includes conversation history when relevant

3. **Response Generation**
   - Uses a LangChain pipeline for response generation
   - Includes both semantic knowledge and episodic memories
   - Maintains conversation context

4. **Memory Storage**
   - Automatically stores interactions as episodic memories
   - Links related memories for better context retention
   - Supports memory updates and consolidation

### Example Query Flow

1. User submits a question
2. System searches across relevant namespaces
3. Retrieves and ranks matching memories
4. Generates a response using the context
5. Stores the interaction for future reference

## License

MIT
