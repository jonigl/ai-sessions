import os

import chromadb
from chromadb.utils import embedding_functions
#from transformers import AutoTokenizer, AutoModel
import ollama

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from pdfFunctions import extract_text_from_pdf_tesseract, extract_text_from_pdf_pypdf
from embedFunctions import generate_embedding, generate_embedding_ollama, generate_embedding_splitting
#from embedFiles import embedDoc

# Configuración de ChromaDB
chroma_client = chromadb.PersistentClient(path=".chroma")
collection = chroma_client.get_or_create_collection(name="pdf_collection2")


ollama_client = ollama.Client

# Directorio con los archivos PDF
pdf_directory = "./pdf"
pdf_text_directory = "./pdf_text"



#for filename in os.listdir(pdf_directory):
#    if filename.endswith(".pdf"):
#        pdf_path = os.path.join(pdf_directory, filename)
#        text = extract_text_from_pdf_tesseract(pdf_path)
        ## Tesseract tiene buena transformación de archivos pdf de imagen
#        embedDoc(text, False, 100, 10, "llama3.2:1b", collection)


# Procesar cada archivo PDF en el directorio
print("Procesando archivos en pdf imagen")
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        text = extract_text_from_pdf_tesseract(pdf_path)
        ## Tesseract tiene buena transformación de archivos pdf de imagen

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        #docs = text_splitter.split_documents(documents=[text])
        docs = text_splitter.split_text(text=text)
        for doc in docs:
            #embedding = generate_embedding_ollama(doc)
            docDictionary = {}
            docDictionary["metadata"] = { 'source': pdf_path }
            docDictionary["page_content"] = doc
            embedding = generate_embedding_splitting(
                docDictionary
            )        
            # Guardar el embedding en ChromaDB
            collection.add(
                documents=[doc],
                embeddings=[embedding],
                ids=[filename]
            )

print("Proceso completado. Los embeddings han sido guardados en ChromaDB.")

print("Procesando archivos en pdf texto")
for filename in os.listdir(pdf_text_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_text_directory, filename)
        text = extract_text_from_pdf_pypdf(pdf_path)
        #embedding = generate_embedding_ollama(text)
        
        chunks, embedding = generate_embedding_splitting(text)
        #docDictionary = {}
        #docDictionary["metadata"] = { 'source': pdf_path }
        #docDictionary["page_content"] = text
        #embedding = generate_embedding_splitting(
        #    docDictionary
        #)

        # Guardar el embedding en ChromaDB
        collection.add(
            documents=chunks,
            embeddings=embedding,
            ids=[filename]
        )

print("Proceso completado. Los embeddings han sido guardados en ChromaDB.")




def queryOllama():
    messages = [
    {
     'role': 'user',
     'content': 'Why is the sky blue?',
    } ,
    ]

    for part in chat('llama3.1:8b', messages=messages, stream=True):
        print(part['message']['content'], end='', flush=True)

def  query_chromadb ( query_text, n_results= 1 ): 
    """ 
    Consulta la colección ChromaDB para encontrar documentos relevantes. 
    
    Args: 
        query_text (str): La consulta de entrada. 
        n_results (int): La cantidad de resultados principales a devolver. 
    
    Devuelve: 
        lista de dict: Los documentos coincidentes principales y sus metadatos. 
    """
    results = collection.query( 
        query_texts=[query_text], 
        n_results=n_results 
    ) 
    return results[ "documents" ], results[ "metadatas" ] 




def queryOllamaEmbedded():
    
    queries = [#"What amount of students have assisted to the exam ?", 
              #  "What amount of students have approved the exam?",
              #  "What are the benefits of representing a general tree structure as a binary tree?",
              #  "How can I represent a general tree as a binary tree, as Knuth indicates?",
              #  "What are the basic differences between trees and binary trees? ",
                "What is the natural correspondance between forests and binary trees?"
               ]
 
    for query_text in queries:
    
        retrieved_docs, metadata = query_chromadb(query_text) 
        context = " " .join(retrieved_docs[0]) if retrieved_docs else  "No relevant docs"

        # Paso 2:
        # Envía la consulta junto con el contexto a Ollama     
        augmented_prompt = f"Using this data: {context}. Respond to this prompt: {query_text}"
        #augmented_prompt = f"Context:{context} \n\nQuery: {query_text}" 
        print ( "######## Augmented Prompt ########" )
        print (f"Context:  {augmented_prompt}\nQuery: {query_text}")


        messages = [
        {
        'role': 'user',
        'content': augmented_prompt,
        } ,
        ]

        for part in ollama.chat('llama3.1:8b', messages=messages, stream=True):
            print(part['message']['content'], end='', flush=True)

        print ( "\n##################################\n" )


queryOllamaEmbedded()
