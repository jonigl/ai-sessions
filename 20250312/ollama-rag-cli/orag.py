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
    
    # Step 2: Split documents into chunks
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
    
    # Step 3: Create embeddings for each chunk
    # Embeddings are vector representations of text that capture semantic meaning
    print(f"Creating embeddings using {embedding_model}...")
    embeddings = OllamaEmbeddings(model=embedding_model)
    
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
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
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
    
    # Step 5: Set up the RAG chain - same as before
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": promptTemplate}
    )
    
    print(f"\nRAG Chat initialized with {llm_model}! Type 'exit' to quit.\n")
    
    # Initialize conversation history list
    # This will store tuples of (question, answer) for context
    history = []
    
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
                current_query = f"{history_context}Based on the above conversation, please answer: {user_input}"
            
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
