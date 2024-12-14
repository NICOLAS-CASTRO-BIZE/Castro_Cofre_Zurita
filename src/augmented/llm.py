from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde .env
load_dotenv()

# Configurar la clave de OpenAI desde el archivo .env
openai_api_key = os.getenv("OPENAI_API_KEY")

# Función para generación aumentada
def generate_augmented_response(query, ranked_docs):
    """Genera una respuesta clara y concisa basada en documentos rankeados."""
    context = " ".join([doc.page_content for doc in ranked_docs])
    
    # Configurar el modelo GPT-4
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # modelo "gpt-4"
        openai_api_key=openai_api_key,
        temperature=0.7  
    )
    
    # Crear mensajes en el formato esperado por LangChain
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions concisely."),
        HumanMessage(content=f"Context: {context}\nQuestion: {query}\nAnswer concisely:")
    ]
    
    # Generar respuesta
    response = llm(messages)
    
    # Devolver el contenido de la respuesta
    return response.content.strip()
