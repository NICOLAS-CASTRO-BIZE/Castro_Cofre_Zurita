
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde .env
load_dotenv()

# Configurar la clave de OpenAI desde el archivo .env
openai_api_key = os.getenv("OPENAI_API_KEY")

def generate_augmented_response(query, ranked_docs):
    # Combine the content of ranked documents
    context = " ".join([doc.page_content for doc in ranked_docs])
    
    # Handle cases with no context
    if not ranked_docs:
        return "I am an assistant for food regulation, and I do not know the answer to that question."
    
    # Configure the LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # Use "gpt-4" if needed
        openai_api_key=openai_api_key,
        temperature=0.05  # Deterministic responses
    )
    
    # Define the messages with clear instructions
    messages = [
        SystemMessage(content="""
            You are an assistant designed to answer questions about a provided context. 
            Do not use any external knowledge or assumptions beyond the provided context. 
            If you do not have enough related information in the context, respond: 'I am an assistant for food regulation, and I do not know the answer to that question.' 
            Keep your responses concise and limit them to a maximum of three sentences.
            The questions could be in any language but the context is in portuguese.
        """),
        HumanMessage(content=f"Context: {context}\nQuestion: {query}\nAnswer in spanish concisely based on the context:")
    ]
    
    # Generate response
    response = llm(messages)
    
    # Return the response content
    return response.content.strip()
