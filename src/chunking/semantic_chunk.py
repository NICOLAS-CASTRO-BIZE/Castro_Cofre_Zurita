from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def generate_semantic_chunk(documents, chunk_size: int = 512,chunk_overlap: int=200) -> list:
    """
    Realiza partición semántica en un texto dado utilizando embeddings.

    Args:
        documents: Lista de documentos procesados (output de PyPDFLoader).
        chunk_size (int): Tamaño máximo de tokens en cada chunk.
    
    Returns:
        List[Document]: Lista de documentos con chunks semánticamente coherentes.
    """
    # seleccionar modelo a utilizar
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Concatenar el texto de los documentos
    all_texts = [doc.page_content for doc in documents]  # Extraer texto de cada documento
    text = " ".join(all_texts)  # Concatenar en una sola cadena

    # Usar RecursiveCharacterTextSplitter para dividir en chunks iniciales
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # Generar embeddings para cada chunk
    embeddings = np.array([embedding_model.embed_query(chunk) for chunk in chunks])
    
    # Calcular similitudes entre embeddings consecutivos
    similarities = cosine_similarity(embeddings)
    
    # Crear chunks semánticos combinando texto basado en similitudes
    semantic_chunks = []
    current_chunk = chunks[0]
    
    for i in range(1, len(chunks)):
        if similarities[i - 1, i] > 0.8:  # Umbral para unir chunks
            current_chunk += " " + chunks[i]
        else:
            semantic_chunks.append(current_chunk)
            current_chunk = chunks[i]
    
    semantic_chunks.append(current_chunk)  # Agregar el último chunk
    
    # Convertir los chunks en objetos Document
    chunk_documents = [Document(page_content=chunk) for chunk in semantic_chunks]
    
    return chunk_documents
