from langchain.ranking import Ranker

def rank_results(query, documents):
    """Reorganiza los documentos recuperados según relevancia."""
    ranker = Ranker(model_name="distilbert-base-uncased")  # selecciona modelo
    ranked_docs = ranker.rank(query=query, documents=documents)
    return ranked_docs