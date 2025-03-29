from transformers import AutoTokenizer, AutoModel
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

TARGET_DIM=768
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

def generate_embedding_splitting(text):

    chunk_size=1000
    chunk_overlap=200
    embedding_model="nomic-embed-text"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Target size of each chunk
        chunk_overlap=chunk_overlap,  # How much chunks should overlap to maintain context
        separators=["\n\n", "\n", ". ", " ", ""]  # Try to split on these separators in order
    )
    #print (text)
    # docs = text_splitter.create_documents([text])
    #print (docs)
    #print (docs.page_content)
    
    chunks = text_splitter.split_text(text)
    embeddings = OllamaEmbeddings(model=embedding_model)

    return chunks, fix_embedding_dim(embeddings.embed_documents(chunks))


# fixes embed dimensions
def fix_embedding_dim(embedding, target_dim=TARGET_DIM):
    i=0
    print(f"Target dim = {TARGET_DIM}")
    for emb in embedding:
        if len(emb) > target_dim:
            return emb[:target_dim]
        elif len(emb) < target_dim:
            return emb + [0.0] * (target_dim - len(emb))
        embedding[i] = emb
        i+=i
    return embedding