#import streamlit as st

#st.title("Hello")

"""Aplicación principal para ejecutar el flujo RAG."""

from src.loaders.load import load_pdf
from src.chunking.chunk import generate_chunks
from src.chunking.semantic_chunk import generate_semantic_chunk
from src.embedding.embedding import generate_embeddings
#from src.vector_store_client.qdrant import create_qdrant_vector_store
from src.vector_store_client.qdrant_hybrid import create_qdrant_vector_store
from src.vector_store_client.hybrid import run_hybrid_search
from src.retrievers.retrieve import retrieve_documents
from src.retrievers.rank import rank_results
from src.augmented.llm import generate_augmented_response
import streamlit as st
#from langchain.globals import set_verbose

# Desactivar verbose
#set_verbose(False)

# Adicional para evitar mensajes de langchain

#from langchain import verbose

# Configurar verbose globalmente
#verbose = False

def main_query(query):
    # Cargar documentos
    file_path = "data/biblioteca-de-alimentos.pdf"
    #"Castro_Cofre_Zurita/data/biblioteca-de-alimentos.pdf"
    documents = load_pdf(file_path)

    # Dividir en chunks
    output_folder = "..data/chunks/"
    #chunks = generate_chunks(documents)
    chunks = generate_semantic_chunk(documents, chunk_size=1000,chunk_overlap=200)
    # Generar embeddings y obtener el modelo de embeddings

    embeddings, embedding_model = generate_embeddings(chunks)

    # Crear base vectorial usando el modelo de embeddings

    #vector_store = create_qdrant_vector_store(chunks, embedding_model)
    #vector_store = create_qdrant_vector_store(chunks, embedding_model)

    # Consultar la base vectorial
    #query = "Quais são as normas para lactantes no Brasil?"
    #"O que fala acerca do rotulagem?"
    #"Quais são as normas para lactantes no Brasil?"

    # devuelve resultados con hybrid search
    #results = retrieve_documents(query, vector_store,2)
    results = run_hybrid_search(chunks, embedding_model, query)

    print("Results passed to Reranker:", [doc.page_content for doc in results])


    # Aplicar ranking a los resultados
    ranked_docs =  rank_results(query, results) #results
    
    # Generar respuesta aumentada
    response = generate_augmented_response(query, ranked_docs)
    
    return response, ranked_docs

    # Mostrar resultados
    
    # for doc in results:
    #     print(doc.page_content)
        
        
    # Display results
    for i, doc in enumerate(results):
        print(f"Result {i + 1}:\n{doc}\n") 
# Interfaz de usuario con Streamlit
def main():
    st.title("Pipeline de RAG alimentos Brasil")
    
    # Entrada de usuario
    query = st.text_input("Introduce tu pregunta sobre la legislacion Brasilera:")
    
    if st.button("Buscar") and query:

        
        with st.spinner("Procesando la consulta..."):
            response, ranked_docs = main_query(query)
        
        # Mostrar respuesta generada
        st.subheader("Respuesta generada:")
        st.write(response)

        # Mostrar resultados
        st.subheader("Resultados recuperados:")
        for i, doc in enumerate(ranked_docs):
            st.write(f"**Resultado {i + 1}:** {doc.page_content[:500]}...")  # Mostrar primeros 500 caracteres
        


if __name__ == "__main__":
    main()  