import os
import sys
import click  # CLI framework for Python
import shutil
from typing import List, Optional
from pathlib import Path
# import Chroma  # Vector database for storing embeddings

# LangChain imports for document handling and RAG system components
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For chunking documents
from langchain_community.document_loaders import (
    TextLoader,       # Loads text files
    PyPDFLoader,      # Loads PDF files
    CSVLoader,        # Loads CSV files
    Docx2txtLoader,   # Loads Word documents
    UnstructuredMarkdownLoader  # Loads Markdown files
)
#from langchain_community.embeddings import OllamaEmbeddings  # Interface to Ollama embedding models # DEPRECATED
from langchain_community.llms import Ollama  # Interface to Ollama language models
# from langchain_community.vectorstores import Chroma  # ChromaDB integration for LangChain # DEPRECATED
from langchain.chains import RetrievalQA  # RAG implementation in LangChain
from langchain.prompts import PromptTemplate  # For creating custom prompts
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma

# Configuration variables - these can be modified as needed
CHROMA_DB_DIR = os.path.expanduser("~/ragdb")  # Where embeddings will be stored
EMBEDDING_MODEL = "nomic-embed-text"  # Default embedding model - good balance of quality and efficiency
LLM_MODEL = "llama3:8b"  # Default LLM - good performance on M2 Mac with 16GB RAM
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
    
    # Step 3: Create embeddings for each chunk
    # Embeddings are vector representations of text that capture semantic meaning
    print(f"Creating embeddings using {embedding_model}...")
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # Step 4: Store the embeddings in a vector database (ChromaDB)
    print(f"Storing embeddings in {CHROMA_DB_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,  # Our document chunks
        embedding=embeddings,  # The embedding function
        persist_directory=CHROMA_DB_DIR  # Where to save the database
    )
    # Make sure to save the database to disk
    # vectorstore.persist() # This is done automatically by ChromaDB
    
    print(f"Successfully embedded {len(chunks)} document chunks into the database.")

@cli.command()
@click.option('--embedding-model', default=EMBEDDING_MODEL, help='Ollama embedding model to use')
@click.option('--llm-model', default=LLM_MODEL, help='Ollama LLM model to use')
@click.option('--k', default=4, help='Number of documents to retrieve')
def chat(embedding_model, llm_model, k):
    """
    Chat with your documents using a RAG-enabled LLM.
    
    This command will:
    1. Load the vector database with your embedded documents
    2. Set up a retrieval system to find relevant document chunks
    3. Connect to an Ollama LLM
    4. Start an interactive chat loop where your questions are answered
       using information from your documents
    """
    # Check if the database exists - we need embedded documents to chat with
    if not os.path.exists(CHROMA_DB_DIR):
        print(f"No database found at {CHROMA_DB_DIR}. Please run 'embed' command first.")
        return
    
    # Step 1: Set up embeddings and vector store
    # We need the same embedding model that was used to create the embeddings
    # to ensure the query vectors are in the same space as the document vectors
    print(f"Loading embeddings with {embedding_model}...")
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # Load the existing vector database from disk
    print("Loading vector database...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings  # This is used to embed the queries
    )
    
    # Step 2: Set up the retriever
    # The retriever is responsible for finding the most relevant document chunks
    # based on the similarity between the query embedding and document embeddings
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Use similarity search (cosine similarity)
        search_kwargs={"k": k}  # Return the k most similar chunks
    )
    
    # Step 3: Set up the Language Model
    # This is the model that will generate responses based on the retrieved context
    print(f"Loading LLM {llm_model}...")
    llm = OllamaLLM(model=llm_model)
    
    # Step 4: Create a RAG prompt template
    # This template formats the context and question for the LLM
    # It's crucial for guiding the LLM to use the retrieved information
    template = """
    Answer the question based only on the following context. If you don't know the answer or 
    the information is not provided in the context, just say so - don't make up an answer. 
    Use the same language as the user question.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    
    # Step 5: Set up the RAG chain
    # This connects the retriever and LLM together into a QA system
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,  # The language model
        chain_type="stuff",  # "stuff" means we stuff all context into one prompt
        retriever=retriever,  # The retriever that finds relevant documents
        chain_type_kwargs={"prompt": prompt}  # Pass our custom prompt
    )
    
    print(f"\nRAG Chat initialized with {llm_model}! Type 'exit' to quit.\n")
    
    # Step 6: Start the interactive chat loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
                
            # Skip empty inputs
            if user_input.strip() == "":
                continue
                
            # Process the query through our RAG pipeline
            print("\nThinking...")
            # The RAG process happens here:
            # 1. Query is embedded
            # 2. Similar documents are retrieved
            # 3. LLM generates a response using the documents
            result = qa_chain.invoke({"query": user_input})
            
            # Display the result
            print(f"\nAssistant: {result['result']}")
            
        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\nGoodbye!")
            break
        except Exception as e:
            # Handle errors
            print(f"Error: {e}")

if __name__ == "__main__":
    cli()
