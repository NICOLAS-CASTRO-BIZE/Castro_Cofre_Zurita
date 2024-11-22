# Step 1: Load and preprocess the PDF
"""Este módulo contiene funciones para cargar documentos PDF utilizando la biblioteca LangChain."""

from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Cargar las variables de entorno
load_dotenv()

def load_pdf(file_path: str):
    """
    Carga un archivo PDF y devuelve su contenido como documentos procesados.

    Args:
        file_path: Ruta al archivo PDF a cargar.

    Returns:
        docs: Lista de documentos procesados extraídos del PDF.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs