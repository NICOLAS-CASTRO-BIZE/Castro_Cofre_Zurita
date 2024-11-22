def generate_chunks(documents, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Divide el texto de un conjunto de documentos en chunks.

    Args:
        documents: Lista de documentos procesados (output de PyPDFLoader).
        chunk_size: Tamaño máximo de cada chunk en caracteres (por defecto, 1000).
        chunk_overlap: Cantidad de caracteres de solapamiento entre chunks (por defecto, 200).

    Returns:
        chunks: Lista de chunks generados a partir del texto de los documentos.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # Concatenar el texto de los documentos
    all_texts = [doc.page_content for doc in documents]  # Extraer texto de cada documento
    concatenated_text = " ".join(all_texts)  # Concatenar en una sola cadena

    # Crear documentos directamente desde el texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.create_documents([concatenated_text])

    # Dividir los documentos en chunks
    chunks = text_splitter.split_documents(documents)
    return chunks