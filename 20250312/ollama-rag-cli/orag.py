import os
import sys
import click  # CLI framework for Python
import shutil
from typing import List, Optional
from pathlib import Path
from chromadb.config import Settings # ChromaDB settings for disabling telemetry

# LangChain imports for document handling and RAG system components
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For chunking documents
from langchain_community.document_loaders import (
    TextLoader,       # Loads text files
    PyPDFLoader,      # Loads PDF files
    CSVLoader,        # Loads CSV files
    Docx2txtLoader,   # Loads Word documents
    UnstructuredMarkdownLoader,  # Loads Markdown files
)
from langchain_community.vectorstores.utils import filter_complex_metadata  # To filter out complex metadata
from langchain_community.llms import Ollama  # Interface to Ollama language models
from langchain.chains import RetrievalQA  # RAG implementation in LangChain
from langchain.prompts import PromptTemplate  # For creating custom prompts
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from prompt_toolkit import prompt
from langchain_unstructured import UnstructuredLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # For streaming responses

# ===============================================================================================
# OVERVIEW OF RAG (Retrieval-Augmented Generation)
# ===============================================================================================
# RAG combines the strengths of retrieval-based systems with generative AI models. The process works as follows:
#
# 1. DOCUMENT INGESTION: Documents are loaded from various file formats (PDFs, text, etc.)
# 2. CHUNKING: Long documents are split into smaller chunks to fit within embedding and token limits
# 3. EMBEDDING: Each chunk is converted into a vector (numerical representation) that captures its semantic meaning
# 4. STORAGE: These vectors are stored in a vector database (ChromaDB in this case)
# 5. RETRIEVAL: When a user asks a question, the question is also converted to a vector and similar document
#    chunks are retrieved through similarity search
# 6. CONTEXT INJECTION: Retrieved relevant documents are injected into the prompt for the LLM
# 7. GENERATION: The LLM generates a response based on both the user question AND the retrieved context
#
# This approach allows the LLM to "know" information it wasn't explicitly trained on, reducing hallucinations
# and enabling more accurate and up-to-date responses from your own document collection.
# ===============================================================================================

# Custom AsciiDoc loader using UnstructuredFileLoader
class AsciiDocLoader(UnstructuredLoader):
    """Load AsciiDoc files using Unstructured."""
    def __init__(self, file_path: str):
        super().__init__(file_path, mode="single")

# Configuration variables - these can be modified as needed
CHROMA_DB_DIR = os.path.expanduser("~/ragdb")  # Where embeddings will be stored
EMBEDDING_MODEL = "nomic-embed-text"  # Default embedding model - good balance of quality and efficiency
LLM_MODEL = "llama3.2:1b"  # Default LLM - good performance on M2 Mac with 16GB RAM
CHUNK_SIZE = 1000  # Size of document chunks in characters
CHUNK_OVERLAP = 200  # Overlap between chunks to maintain context

# Why these defaults?
# - Smaller chunk sizes (1000 chars) help with more precise retrieval
# - Chunk overlaps prevent context loss at chunk boundaries
# - The nomic-embed-text model offers good performance for local embedding
# - llama3.2:1b strikes a balance between quality and resource usage for local deployment

# Map file extensions to the appropriate document loaders
# This allows our application to handle various file formats
LOADER_MAPPING = {
    ".txt": TextLoader,         # Plain text files
    ".pdf": PyPDFLoader,        # PDF documents
    ".csv": CSVLoader,          # CSV data files
    ".docx": Docx2txtLoader,    # Word documents
    ".md": UnstructuredMarkdownLoader,  # Markdown files
    ".adoc": AsciiDocLoader,  # AsciiDoc files
}

# RAG STEP 1: DOCUMENT LOADING
# Different document formats require different loaders to extract text properly
# For example, PDF loading requires parsing the PDF structure, while markdown
# needs to handle special formatting characters

def get_loader_for_file(file_path: str):
    """
    Get the appropriate document loader based on file extension.
    
    This function examines the file extension and returns the corresponding
    LangChain document loader that can properly extract text from that file type.
    
    Args:
        file_path: Path to the file that needs to be loaded
        
    Returns:
        An instantiated LangChain document loader for the given file
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension in LOADER_MAPPING:
        return LOADER_MAPPING[file_extension](file_path)
    else:
        # Default to text loader for unknown types
        print(f"Warning: Unknown file type {file_extension}. Using default TextLoader.")
        return TextLoader(file_path)

def clear_database():
    """
    Clear the existing vector database.
    
    This function removes the ChromaDB directory if it exists and creates a new empty one.
    Used when you want to start fresh with new embeddings without the old ones.
    """
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
        print(f"Cleared existing database at {CHROMA_DB_DIR}")
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

@click.group()
def cli():
    """
    RAG CLI tool for document embedding and chatting.
    
    This is the main entry point for the CLI application.
    Click uses this function as the command group that contains subcommands.
    """
    pass

@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--clear', is_flag=True, help='Clear existing database before embedding')
@click.option('--chunk-size', default=CHUNK_SIZE, help='Size of document chunks')
@click.option('--chunk-overlap', default=CHUNK_OVERLAP, help='Overlap between chunks')
@click.option('--embedding-model', default=EMBEDDING_MODEL, help='Ollama embedding model to use')
def embed(directory, clear, chunk_size, chunk_overlap, embedding_model):
    """
    Embed documents from the given directory.
    
    This command will:
    1. Find all supported files in the directory (recursively)
    2. Load and process each document
    3. Split documents into chunks
    4. Generate embeddings for each chunk
    5. Store embeddings in a vector database
    """
    # Clear the database if requested (useful for starting fresh)
    if clear:
        clear_database()
    
    # ===============================================================================================
    # RAG STEP 1: DOCUMENT COLLECTION
    # ===============================================================================================
    # First, we need to gather all documents from the user's specified directory
    # This includes recursively finding files with supported formats
    # Each document is loaded with its specific loader to extract the text content
    # ===============================================================================================
    
    # Collect all the documents from the directory
    documents = []
    directory_path = Path(directory)
    
    # Walk through the directory recursively (.rglob) to find all files
    for file_path in directory_path.rglob("*"):
        # Check if it's a file and if we have a loader for this file type
        if file_path.is_file() and file_path.suffix.lower() in LOADER_MAPPING:
            try:
                print(f"Loading {file_path}")
                # Get the right loader for this file type
                loader = get_loader_for_file(str(file_path))
                # Load the document - this extracts the text and metadata
                file_docs = loader.load()
                # Add to our collection of documents
                documents.extend(file_docs)
                print(f"Loaded {len(file_docs)} document(s) from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    # Check if we found any documents
    if not documents:
        print("No documents found or loaded. Exiting.")
        return
    
    print(f"Loaded {len(documents)} document(s) in total.")
    
    # ===============================================================================================
    # RAG STEP 2: DOCUMENT CHUNKING
    # ===============================================================================================
    # Why chunk documents?
    # 1. Embedding models have input token limits
    # 2. Smaller chunks improve retrieval precision
    # 3. Retrieval works best with focused, coherent text segments
    #
    # The RecursiveCharacterTextSplitter is smart - it tries to break at natural boundaries
    # like paragraphs first, then sentences, etc., to keep semantic coherence in chunks
    # ===============================================================================================
    
    # The RecursiveCharacterTextSplitter is a smart splitter that tries to keep
    # related text together by using a hierarchy of separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Target size of each chunk
        chunk_overlap=chunk_overlap,  # How much chunks should overlap to maintain context
        separators=["\n\n", "\n", ". ", " ", ""]  # Try to split on these separators in order
    )
    
    print("Splitting documents into chunks...")
    # This creates smaller, manageable chunks from our documents
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    
    # Filter complex metadata to prevent ChromaDB errors (those error happened when using adoc files)
    print("Filtering complex metadata...")
    for chunk in chunks:
        # Process each metadata field manually
        filtered_metadata = {}
        for key, value in chunk.metadata.items():
            # Only keep simple data types
            if isinstance(value, (str, int, float, bool)):
                filtered_metadata[key] = value
            elif isinstance(value, list) and len(value) > 0:
                # Convert lists to string if possible
                filtered_metadata[key] = str(value)
        # Replace with filtered metadata
        chunk.metadata = filtered_metadata
    
    # ===============================================================================================
    # RAG STEP 3: EMBEDDING GENERATION
    # ===============================================================================================
    # Embedding is the process of converting text into numerical vectors
    # These vectors represent the semantic meaning of the text in a high-dimensional space
    # Similar texts will have similar vector representations (close in the vector space)
    #
    # This is the heart of how RAG works - when a question is asked, we convert it to a vector
    # and find the document chunks with the most similar vectors
    # ===============================================================================================
    
    # Step 3: Create embeddings for each chunk
    # Embeddings are vector representations of text that capture semantic meaning
    print(f"Creating embeddings using {embedding_model}...")
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # ===============================================================================================
    # RAG STEP 4: VECTOR DATABASE STORAGE
    # ===============================================================================================
    # ChromaDB is a vector database that stores:
    # 1. The document chunks themselves (text)
    # 2. The vector embeddings for each chunk
    # 3. Metadata about the documents (source, page numbers, etc.)
    #
    # When stored, these vectors can be efficiently searched to find the most relevant
    # information for any query. The database uses approximate nearest neighbor algorithms
    # to quickly find similar vectors without having to check every single one.
    # ===============================================================================================
    
    # Step 4: Store the embeddings in a vector database (ChromaDB)
    print(f"Storing embeddings in {CHROMA_DB_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,  # Our document chunks
        embedding=embeddings,  # The embedding function
        persist_directory=CHROMA_DB_DIR,  # Where to save the database
        client_settings=Settings(anonymized_telemetry=False)  # Disable telemetry
    )
    # Make sure to save the database to disk
    # vectorstore.persist() # This is done automatically by ChromaDB
    
    print(f"Successfully embedded {len(chunks)} document chunks into the database.")

@cli.command()
@click.option('--embedding-model', default=EMBEDDING_MODEL, help='Ollama embedding model to use')
@click.option('--llm-model', default=LLM_MODEL, help='Ollama LLM model to use')
@click.option('--k', default=4, help='Number of documents to retrieve')
@click.option('--stream/--no-stream', default=True, help='Stream responses from the LLM')
@click.option('--keep-history/--no-history', default=True, help='Maintain conversation history')
@click.option('--history-size', default=3, help='Number of previous exchanges to include in context')
def chat(embedding_model, llm_model, k, stream, keep_history, history_size):
    """
    Chat with your documents using a RAG-enabled LLM with conversation history.
    
    This command will:
    1. Load the vector database with your embedded documents
    2. Set up a retrieval system to find relevant document chunks
    3. Connect to an Ollama LLM
    4. Start an interactive chat loop where your questions are answered
       using information from your documents and previous conversation context
    """
    # Check if the database exists - we need embedded documents to chat with
    if not os.path.exists(CHROMA_DB_DIR):
        print(f"No database found at {CHROMA_DB_DIR}. Please run 'embed' command first.")
        return
    
    # ===============================================================================================
    # RAG STEP 5: SETTING UP RETRIEVAL
    # ===============================================================================================
    # Here we load the vector database and create a retriever
    # The retriever is responsible for:
    # 1. Converting the user query to a vector using the same embedding model
    # 2. Finding the closest matching document chunks in the vector database
    # 3. Returning the most relevant chunks for context
    #
    # The 'k' parameter controls how many chunks to retrieve - more chunks means 
    # more context but might include less relevant information
    # ===============================================================================================
    
    # Step 1: Set up embeddings and vector store - same as before
    print(f"Loading embeddings with {embedding_model}...")
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    print("Loading vector database...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR, 
        embedding_function=embeddings,
        client_settings=Settings(anonymized_telemetry=False)
    )
    
    # Step 2: Set up the retriever - same as before
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Use similarity search to find the most relevant chunks
        search_kwargs={"k": k}     # Number of chunks to retrieve per query
    )
    
    # ===============================================================================================
    # RAG STEP 6: LANGUAGE MODEL SETUP
    # ===============================================================================================
    # The LLM is the generative part of RAG. It takes:
    # 1. The user's question
    # 2. The retrieved document chunks as context
    # and generates a coherent, informed response
    #
    # Streaming allows the user to see the response as it's being generated
    # rather than waiting for the entire response to complete
    # ===============================================================================================
    
    # Step 3: Set up the Language Model - same as before
    print(f"Loading LLM {llm_model}...")
    
    callbacks = []
    if stream:
        callbacks.append(StreamingStdOutCallbackHandler())
    
    llm = OllamaLLM(
        model=llm_model,
        callbacks=callbacks if stream else None,
        streaming=stream
    )
    
    # ===============================================================================================
    # RAG STEP 7: PROMPT ENGINEERING
    # ===============================================================================================
    # The prompt template is crucial for RAG - it combines:
    # 1. The retrieved document context
    # 2. The user's question
    # 3. Instructions for how to answer (e.g., "only use the provided context")
    #
    # A good RAG prompt clearly separates context from the question and gives
    # clear instructions to the LLM about how to use the context
    # ===============================================================================================
    
    # Step 4: Create the RAG prompt template - we keep this simple
    # This template focuses on the current question and context
    template = """
    Answer the question based only on the following context. If you don't know the answer or 
    the information is not provided in the context, just say so - don't make up an answer. 
    Use the same language as the user question.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    promptTemplate = PromptTemplate.from_template(template)
    
    # ===============================================================================================
    # RAG STEP 8: CHAIN ASSEMBLY
    # ===============================================================================================
    # The RetrievalQA chain combines all the RAG components:
    # 1. It takes a user question
    # 2. Passes it to the retriever to get relevant context
    # 3. Constructs a prompt with the question and retrieved context
    # 4. Sends the prompt to the LLM for final answer generation
    #
    # The "stuff" chain type means we simply "stuff" all retrieved documents into one prompt
    # Other chain types like "refine" or "map_reduce" exist for handling larger amounts of context
    # ===============================================================================================
    
    # Step 5: Set up the RAG chain - same as before
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Simply insert all retrieved docs into one prompt
        retriever=retriever,
        chain_type_kwargs={"prompt": promptTemplate}
    )
    
    print(f"\nRAG Chat initialized with {llm_model}! Type 'exit' to quit.\n")
    
    # Initialize conversation history list
    # This will store tuples of (question, answer) for context
    history = []
    
    # ===============================================================================================
    # RAG STEP 9: CONVERSATION LOOP WITH HISTORY
    # ===============================================================================================
    # Adding conversation history to RAG has several benefits:
    # 1. It maintains context across multiple turns
    # 2. It allows follow-up questions without restating the entire context
    # 3. It enables the LLM to provide more coherent and consistent responses
    #
    # However, history also consumes tokens, so we need to limit how much history we keep
    # ===============================================================================================
    
    # Step 6: Start the interactive chat loop with history tracking
    while True:
        try:
            user_input = prompt("\nYou: ")            
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
                
            # Skip empty inputs
            if user_input.strip() == "":
                continue
                
            # Process the query through our RAG pipeline
            print("\nThinking...")
            
            # Prepare the current query - this is where we add history context
            current_query = user_input
            if keep_history and history:
                # Enhance the query with previous conversation context
                # This helps the LLM understand the current question in light of previous exchanges
                history_context = ""
                for i, (q, a) in enumerate(history[-history_size:]):
                    history_context += f"Previous question {i+1}: {q}\n"
                    history_context += f"Previous answer {i+1}: {a}\n\n"
                
                # Append history to the current query to create continuity
                current_query = f"{history_context} Based on the above conversation, please answer: {user_input}"
            
            # ===============================================================================================
            # RAG EXECUTION FLOW:
            # 1. User question is converted to a vector
            # 2. Vector database is searched for similar document chunks
            # 3. Retrieved chunks are combined with the question in a prompt
            # 4. LLM generates an answer based on the question and provided context
            # 5. Response is streamed back to the user
            # 6. The exchange is stored in history for future context
            # ===============================================================================================
            
            # Execute the RAG query with our possibly enhanced query
            if stream:
                print("\nAssistant: ", end="", flush=True)
                result = qa_chain.invoke({"query": current_query})
                assistant_response = result['result']
                print("\n")  # Add newline after response
            else:
                result = qa_chain.invoke({"query": current_query})
                assistant_response = result['result']
                print(f"\nAssistant: {assistant_response}")
            
            # Store this exchange in our conversation history
            if keep_history:
                history.append((user_input, assistant_response))
                # Limit history size to prevent context bloat
                # This is important for both token limits and relevance
                if len(history) > history_size * 2:  # Keep double the displayed history for potential future uses
                    history = history[-(history_size * 2):]
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()  # Detailed error for debugging

if __name__ == "__main__":
    cli()
