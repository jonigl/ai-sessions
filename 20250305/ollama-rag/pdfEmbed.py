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


def add_text_to_collection(text, pdf_path, collection):
    chunks, embedding = generate_embedding_splitting(text)
    ids = []
    for i in range(len(chunks)):
        ids.append(f"{pdf_path}_{i}")
    # Guardar el embedding en ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embedding,
        ids=ids
    )
# Configuración de ChromaDB
chroma_client = chromadb.PersistentClient(path=".chroma")
collection = chroma_client.get_or_create_collection(name="pdf_collection2")

print(collection)

#exit()

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
        add_text_to_collection(text, pdf_path, collection)


print("Proceso completado. Los embeddings han sido guardados en ChromaDB.")

print("Procesando archivos en pdf texto")
for filename in os.listdir(pdf_text_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_text_directory, filename)
        text = extract_text_from_pdf_pypdf(pdf_path)
        #embedding = generate_embedding_ollama(text)
        add_text_to_collection(text, pdf_path, collection)

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
    query_embedding = generate_embedding_splitting(query_text)

    results = collection.query( 
        query_embeddings=query_embedding[1], 
        n_results=n_results 
    ) 
    return results[ "documents" ], results[ "metadatas" ] 




def queryOllamaEmbedded():
    
    queries = [#"What amount of students have assisted to the exam ?", 
                "What amount of students have approved the exam?",
              #  "What are the benefits of representing a general tree structure as a binary tree?",
              #  "How can I represent a general tree as a binary tree, as Knuth indicates?",
                "What are the basic differences between trees and binary trees? ",
              #  "What is the natural correspondance between forests and binary trees?"
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
