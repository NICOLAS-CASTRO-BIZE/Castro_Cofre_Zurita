
"""Aplicación principal para ejecutar el flujo RAG."""

from src.loaders.load import load_pdf
from src.loaders.clean import clean_text
from src.chunking.semantic_chunk import generate_semantic_chunk
from src.embedding.embedding import generate_embeddings
from src.vector_store_client.hybrid import run_hybrid_search
from src.retrievers.rank import rank_results
from src.augmented.llm import generate_augmented_response
from src.augmented.translator import translate_query
import streamlit as st
#from src.retrievers.retrieve import retrieve_documents
#from src.vector_store_client.qdrant import create_qdrant_vector_store
#from src.vector_store_client.qdrant_hybrid import create_qdrant_vector_store
#from src.chunking.chunk import generate_chunks

def main_query(query):
    # Cargar documentos
    file_path = "data/biblioteca-de-alimentos.pdf"
    #"Castro_Cofre_Zurita/data/biblioteca-de-alimentos.pdf"
    documents = load_pdf(file_path)

    # Aplicar limpieza a cada documento
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    # Dividir en chunks
    output_folder = "..data/chunks/"
    #chunks = generate_chunks(documents)
    chunks = generate_semantic_chunk(documents, chunk_size=200,chunk_overlap=50)
    # Generar embeddings y obtener el modelo de embeddings

    embeddings, embedding_model = generate_embeddings(chunks)

    # devuelve resultados con hybrid search
    results = run_hybrid_search(chunks, embedding_model, query)

    print("Results passed to Reranker:", [doc.page_content for doc in results])

    # Aplicar ranking a los resultados
    ranked_docs =  rank_results(query, results) #results
    
    # Generar respuesta aumentada
    response = generate_augmented_response(query, ranked_docs)
    
    return response, ranked_docs

    # Mostrar resultados

    for i, doc in enumerate(results):
        print(f"Result {i + 1}:\n{doc}\n") 

# Interfaz de usuario con Streamlit
def main():
    st.title("Pipeline de RAG alimentos Brasil")
    
    # Entrada de usuario
    query = st.text_input("Introduce tu pregunta sobre la legislacion Brasilera:")
    
    if st.button("Buscar") and query:

        
        with st.spinner("Procesando la consulta..."):
            # Traducir la consulta al portugués
            query_translated = translate_query(query)
        
        with st.spinner("Buscando en documentos..."):
            # Enviar la consulta traducida al pipeline principal
            response, ranked_docs = main_query(query_translated)
          
        # Mostrar respuesta generada
        st.subheader("Respuesta generada:")
        st.write(response)

        # Mostrar resultados
        st.subheader("Resultados recuperados:")
        for i, doc in enumerate(ranked_docs):
            st.write(f"**Resultado {i + 1}:** {doc.page_content[:800]}...")  # Mostrar primeros 800 caracteres
        


if __name__ == "__main__":
    main()  

#Ejecutar 
# streamlit run main_app_final.py