"""Este módulo contiene funciones para crear y gestionar una base de datos vectorial utilizando Qdrant."""

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Cargar las variables de entorno
load_dotenv()

def create_qdrant_vector_store(docs: list, embedding_model, collection_name: str = "demo_collection", path: str = "/tmp/langchain_qdrant", vector_size: int = 384) -> QdrantVectorStore:
    """
    Crea una base de datos vectorial en Qdrant y almacena documentos con sus embeddings.

    Args:
        docs: Lista de documentos (chunks) a almacenar en la base de datos.
        embedding_model: Modelo de embeddings a utilizar para calcular vectores.
        collection_name: Nombre de la colección en Qdrant (por defecto, "demo_collection").
        path: Ruta donde se almacenará la base de datos Qdrant (por defecto, "/tmp/langchain_qdrant").
        vector_size: Tamaño de los vectores de embedding (por defecto, 768).

    Returns:
        qdrant_store: Objeto QdrantVectorStore listo para realizar búsquedas.
    """
    # Inicializar el cliente Qdrant
    client = QdrantClient(path=path)

    # Eliminar la colección existente si ya existe
    client.delete_collection(collection_name=collection_name)

    # Crear una nueva colección en Qdrant
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # Crear un QdrantVectorStore
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )

    # Agregar documentos a la colección
    vector_store.add_documents(docs)

    print(f"Base de datos vectorial '{collection_name}' creada y {len(docs)} documentos añadidos.")

    return vector_store