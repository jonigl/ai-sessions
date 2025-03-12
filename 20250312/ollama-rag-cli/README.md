# orag a python RAG CLI

A command-line application for Retrieval-Augmented Generation (RAG) that allows you to chat with your personal documents using local language models through Ollama.

## Overview

This tool enables you to:

1. **Embed** documents from a directory into a vector database
2. **Chat** with an LLM that can retrieve and use information from your documents

The application is designed to work on systems with limited resources (like Macs with 16GB RAM) by leveraging Ollama's efficient local AI models.

## How RAG Works

Retrieval-Augmented Generation combines document retrieval with language model generation:

1. **Document Processing**:
   - Documents are split into smaller chunks
   - Each chunk is converted to a vector embedding that captures its semantic meaning
   - These embeddings are stored in a vector database (ChromaDB)

2. **Query Processing**:
   - Your question is converted to an embedding using the same model
   - The system finds document chunks with similar embeddings to your question
   - These relevant chunks are retrieved

3. **Response Generation**:
   - The retrieved document chunks and your question are sent to an LLM
   - The LLM generates a response based on the context from your documents

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Sufficient disk space for document storage and embeddings

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/document-rag-cli.git
   cd document-rag-cli
   ```

2. Install the required dependencies:
   ```bash
   pip install langchain-community chromadb click pydantic unstructured pypdf
   ```

3. Install Ollama models:
   ```bash
   # Install the recommended embedding model
   ollama pull nomic-embed-text
   
   # Install at least one LLM
   ollama pull llama3:8b
   # Or for lighter resource usage
   ollama pull phi3:3.8b
   ```

## Usage

### Embedding Documents

To embed documents from a directory:

```bash
python orag.py embed /path/to/your/documents
```

Options:
- `--clear`: Clear existing database before embedding
- `--chunk-size`: Set size of document chunks (default: 1000)
- `--chunk-overlap`: Set overlap between chunks (default: 200)
- `--embedding-model`: Specify Ollama embedding model (default: nomic-embed-text)

Example:
```bash
python orag.py embed ~/Documents/research --clear --chunk-size 800 --embedding-model all-MiniLM-L6-v2
```

### Chatting with Documents

To start a chat session with your documents:

```bash
python orag.py chat
```

Options:
- `--embedding-model`: Specify Ollama embedding model (default: nomic-embed-text)
- `--llm-model`: Specify Ollama LLM model (default: llama3:8b)
- `--k`: Number of document chunks to retrieve per query (default: 4)

Example:
```bash
python orag.py chat --llm-model phi3:3.8b --k 6
```

## Supported File Types

The tool currently supports:
- Plain text files (.txt)
- PDF documents (.pdf)
- Word documents (.docx)
- CSV files (.csv)
- Markdown files (.md)

## Configuration

Default settings are defined at the top of the script:

```python
CHROMA_DB_DIR = os.path.expanduser("~/ragdb")  # Database location
EMBEDDING_MODEL = "nomic-embed-text"           # Default embedding model
LLM_MODEL = "llama3:8b"                        # Default LLM model
CHUNK_SIZE = 1000                              # Default chunk size
CHUNK_OVERLAP = 200                            # Default chunk overlap
```

## Recommended Models for Mac M2 with 16GB RAM

### Embedding Models:
- **nomic-embed-text** (default): Best balance of quality and efficiency
- **all-MiniLM-L6-v2**: Lighter option, still good quality
- **e5-small-v2**: Another lightweight option

### LLM Models:
- **llama3:8b** (default): Best quality for the resource constraints
- **mistral:7b**: Good alternative with different capabilities
- **phi3:3.8b**: Lighter option if you're experiencing memory pressure

## Technical Details

### Key Components:

1. **Document Loading**: Uses LangChain's document loaders to extract text from various file formats.

2. **Text Splitting**: Implements recursive character text splitting with configurable chunk size and overlap.

3. **Embedding Generation**: Uses Ollama to create semantic vector embeddings for each text chunk.

4. **Vector Storage**: Stores embeddings in ChromaDB for efficient similarity search.

5. **Retrieval**: Retrieves relevant document chunks based on query similarity.

6. **Response Generation**: Uses the Ollama LLM to generate responses based on retrieved context.

## Extending the Application

### Adding Support for New File Types

To add support for additional file types, update the `LOADER_MAPPING` dictionary:

```python
LOADER_MAPPING = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    # Add new mappings here
    ".new_extension": NewLoader,
}
```

### Using Different Vector Databases

The application uses ChromaDB by default, but you can modify it to use other vector databases supported by LangChain, such as FAISS, Pinecone, or Milvus.

## Troubleshooting

### Memory Issues
- If you encounter memory errors, try using smaller models like phi3:3.8b
- Reduce the chunk size and number of retrievals (k)
- Run the embedding and chat commands separately (don't run both simultaneously)

### Model Loading Errors
- Ensure Ollama is running (`ollama serve`)
- Verify that you've pulled the required models
- Check Ollama logs for detailed error messages

## License

[MIT License](LICENSE)
