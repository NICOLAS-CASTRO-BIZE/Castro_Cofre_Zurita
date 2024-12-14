from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

def run_hybrid_search(docs, embedding_model, query, sparse_model="Qdrant/bm25", location=":memory:", collection_name="hybrid_search_collection", top_k=5):
    """
    Realiza una búsqueda híbrida en documentos utilizando Qdrant.

    Args:
        docs (list): Lista de documentos para cargar en el vector store.
        embedding_model: Modelo de embeddings para calcular vectores densos.
        query (str): La consulta que se quiere buscar.
        sparse_model (str): Modelo para embeddings dispersos (por defecto "Qdrant/bm25").
        location (str): Ubicación del almacenamiento ("memory" para almacenamiento temporal).
        collection_name (str): Nombre de la colección Qdrant.
        top_k (int): Número de documentos a recuperar.

    Returns:
        list: Documentos encontrados que corresponden a la consulta.
    """
    # Crear sparse embeddings (dispersos)
    sparse_embeddings = FastEmbedSparse(model_name=sparse_model)

    # Crear el vector store en memoria con soporte para búsquedas híbridas
    qdrant = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embedding_model,          # Embeddings densos
        sparse_embedding=sparse_embeddings, # Embeddings dispersos
        location=location,                  # ":memory:" para evitar persistencia
        collection_name=collection_name,
        retrieval_mode=RetrievalMode.HYBRID # Activar modo híbrido
    )

    # Realizar la búsqueda híbrida
    found_docs = qdrant.similarity_search(query, k=top_k)

    return found_docs
