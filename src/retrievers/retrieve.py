def retrieve_documents(query: str, vector_store, k: int = 5) -> list:
    """
    Recupera los documentos más relevantes para una consulta dada utilizando QdrantVectorStore.

    Args:
        query: Texto de la consulta.
        vector_store: Objeto QdrantVectorStore donde se encuentran almacenados los documentos.
        k: Número de documentos relevantes a recuperar (por defecto, 5).

    Returns:
        docs: Lista de los documentos más relevantes encontrados para la consulta.
    """
    # Realizar la búsqueda de similitud en el vector store
    docs = vector_store.similarity_search(query, k=k)

    return docs