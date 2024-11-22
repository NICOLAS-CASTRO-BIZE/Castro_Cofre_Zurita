#import streamlit as st

#st.title("Hello")

"""Aplicación principal para ejecutar el flujo RAG."""

from src.loaders.load import load_pdf
from src.chunking.chunk import generate_chunks
from src.embedding.embedding import generate_embeddings
from src.vector_store_client.qdrant import create_qdrant_vector_store
from src.retrievers.retrieve import retrieve_documents
from src.extras.clear_terminal import clear_terminal
from src.extras.suppress_output import SuppressOutput

#from langchain.globals import set_verbose

# Desactivar verbose
#set_verbose(False)

# Adicional para evitar mensajes de langchain

#from langchain import verbose

# Configurar verbose globalmente
#verbose = False

def main():
    # Cargar documentos
    file_path = "data/biblioteca-de-alimentos.pdf"
    #"Castro_Cofre_Zurita/data/biblioteca-de-alimentos.pdf"
    documents = load_pdf(file_path)

    # Dividir en chunks
    output_folder = "..data/chunks/"
    chunks = generate_chunks(documents)

    # Generar embeddings y obtener el modelo de embeddings
    with SuppressOutput():
        embeddings, embedding_model = generate_embeddings(chunks)

    # Crear base vectorial usando el modelo de embeddings
    with SuppressOutput():
        vector_store = create_qdrant_vector_store(chunks, embedding_model)

    # Consultar la base vectorial
    query = "Quais são as normas para lactantes no Brasil?"
    #"O que fala acerca do rotulagem?"
    #"Quais são as normas para lactantes no Brasil?"
    with SuppressOutput():
        results = retrieve_documents(query, vector_store,2)

    # Llamar a la función para limpiar el terminal
    clear_terminal()
    
    # Mostrar resultados
    
    # for doc in results:
    #     print(doc.page_content)
        
        
    # Display results
    for i, doc in enumerate(results):
        print(f"Result {i + 1}:\n{doc}\n") 
        
        

if __name__ == "__main__":
    main()
