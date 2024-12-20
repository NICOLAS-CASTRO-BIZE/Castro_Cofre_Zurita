"""
from langchain.ranking import Ranker

def rank_results(query, documents):
    
    ranker = Ranker(model_name="distilbert-base-uncased")  # selecciona modelo
    ranked_docs = ranker.rank(query=query, documents=documents)
    return ranked_docs
"""
from rerankers import Reranker
from langchain.schema import Document
"""
def rank_results(query, retrieved_docs):
    #reranker = Reranker('cross-encoder', verbose=0,model_type='cross-encoder')

    reranker = Reranker("colbert", verbose=0)
    retrieved_docs = [doc.page_content for doc in retrieved_docs]
    reranked_docs = reranker.rank(query, retrieved_docs)
    return reranked_docs
"""


from langchain.schema import Document

def rank_results(query, retrieved_docs):
    """
    Reorganiza los documentos recuperados según relevancia usando un Reranker.
    """
    #reranker = Reranker("colbert", verbose=0,lang='pt')
    reranker = Reranker("cross-encoder", verbose=0,lang='pt')
    # Extraer el contenido de los documentos recuperados
    retrieved_texts = [doc.page_content for doc in retrieved_docs]
    
    # Re-rankear los documentos
    reranked_results = reranker.rank(query, retrieved_texts)
    
    # Verificar las puntuaciones de los documentos
    for i, result in enumerate(reranked_results):
        print(f"Rank {i + 1}: {result.document.text} | Score: {result.score}")

    # Reconstruir los documentos basados en los textos re-rankeados
    reranked_docs = [
        Document(page_content=result.document.text, metadata={})
        for result in reranked_results
    ]

    return reranked_docs
