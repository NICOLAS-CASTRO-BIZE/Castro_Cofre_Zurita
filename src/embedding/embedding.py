# Step 3: Generate embeddings using Hugging Face
"""Este módulo contiene funciones para generar embeddings a partir de texto utilizando Hugging Face."""


def generate_embeddings(chunks: list, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> tuple:
    """
    Genera embeddings a partir de una lista de chunks de texto utilizando un modelo de Hugging Face.

    Args:
        chunks: Lista de objetos Document o cadenas de texto.
        model_name: Nombre del modelo preentrenado de Hugging Face a utilizar.

    Returns:
        embeddings: Lista de vectores de embeddings generados para los chunks de texto.
        embedding_model: El modelo de embeddings utilizado.
    """
    from langchain.embeddings import HuggingFaceEmbeddings

    # Cargar el modelo de embeddings
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    
    # Extraer el texto si los chunks son objetos Document
    texts = [chunk.page_content if hasattr(chunk, 'page_content') else chunk for chunk in chunks]

    # Generar embeddings para los textos
    embeddings = embedding_model.embed_documents(texts)
    
    return embeddings, embedding_model
