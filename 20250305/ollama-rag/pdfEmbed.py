import os
import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModel
import ollama
import pytesseract
from pdf2image import convert_from_path

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Configuración de ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="pdf_collection2")


ollama_client = ollama.Client

# Directorio con los archivos PDF
pdf_directory = "./pdf"
pdf_text_directory = "./pdf_text"





# Función para extraer texto de un archivo PDF
def extract_text_from_pdf_tesseract(pdf_path):
    print(f'Processing {pdf_path}')

    text = convertTesseract(pdf_path)
    return text 



def extract_text_from_pdf_pypdf(pdf_path):
    print(f'Processing {pdf_path}')
    text=""
    with open(pdf_path, "rb") as file:
      reader = PyPDF2.PdfFileReader(file)
      for page_num in range(reader.numPages):
          page = reader.getPage(page_num)
          text += page.extract_text()
    return text
    

def convertTesseract(path):
  # convert to image using resolution 600 dpi 
  pages = convert_from_path(path, 600)

  # extract text
  text_data = ''
  for page in pages:
    text = pytesseract.image_to_string(page)
    text_data += text + '\n'
  #print(text_data)
  return text_data

# Función para generar embeddings usando OpenLLaMA local
def generate_embedding(text):

    tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3.2_1b")
    model = AutoModel.from_pretrained("openlm-research/open_llama_3.2_1b")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    
    # Usamos el último estado oculto como embedding
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embedding



def generate_embedding_ollama(text):
    #tokens = ollama_client.tokenize(text)
    resp = ollama.embed(model='all-minilm', input=text)
    return resp["embeddings"][0]



# Procesar cada archivo PDF en el directorio
print("Procesando archivos en pdf imagen")
for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, filename)
        text = extract_text_from_pdf_tesseract(pdf_path)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        #docs = text_splitter.split_documents(documents=[text])
        docs = text_splitter.split_text(text=text)
        for doc in docs:
            embedding = generate_embedding_ollama(doc)        
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
        embedding = generate_embedding_ollama(text)
        
        # Guardar el embedding en ChromaDB
        collection.add(
            documents=[text],
            embeddings=[embedding],
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
                "What are the benefits of representing a general tree structure as a binary tree?",
                "How can I represent a general tree as a binary tree, as Knuth indicates?",
                "What are the basic differences between trees and binary trees? ",
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
