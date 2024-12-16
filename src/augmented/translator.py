from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

def translate_query(query):
    """
    Utiliza GPT-3.5 Turbo para traducir una consulta al portugués.
    
    Args:
        query (str): Consulta en español o inglés.
    
    Returns:
        str: Consulta traducida al portugués.
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    messages = [
        SystemMessage(content="You are a translation assistant. Translate the following query to Portuguese."),
        HumanMessage(content=f"Translate this to Portuguese: {query}")
    ]
    response = llm(messages)
    return response.content.strip()