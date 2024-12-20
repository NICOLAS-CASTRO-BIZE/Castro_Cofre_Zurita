## End-to-End Retrieval-Augmented Generation, Equipo legislación de Brasil

Welcome to the **Building an End-to-End Retrieval-Augmented Generation System** repository. This repository is designed to guide you through the process of creating a complete Retrieval-Augmented Generation (RAG) system from scratch, following a structured curriculum.

# Alumnos:

Nicolás Oscar Castro Bize,
Carlos Alberto Zurita Olea,
María Ignacia Cofré Poblete

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/CarlosCaris/practicos-rag.git
2. Create a virtual environment
    ```bash
    python -m vevn .venv
3. Activate the environment
   ```bash
    # On Mac
    .venv/bin/activate
    # On Windows
    .venv\Scripts\activate
4. Install requirements
    ```bash
    pip install -r requirements.txt
5. Download and be sure to have file "biblioteca_de_alimentos.pdf" available in:
   ```bash
    data/biblioteca_de_alimentos.pdf
6. Make sure to have a key in .env file as:
   ```bash
    OPENAI_API_KEY="your_api_key"
7.  for running streamlit app please execute on console
    ```bash
    streamlit run main_app_final.py
8. The app will open in a new tab. Then make questions about brazilian food legislation

## Arquitectura de la solución RAG implementada

<img width="772" alt="image" src="https://github.com/user-attachments/assets/f325bb5d-c37a-410b-a545-6111c6536ae7" />

# Consideraciones / español

1) Descargar  o clonar repositorio desde el main
2) Instalar requirements.txt con las librerias utilizadas
3) El usuario debe tener una key de openAI en un archivo .env
4) Guardar o asegurar tener archivo biblioteca_de_alimentos.pdf disponible en data/biblioteca-de-alimentos.pdf
5) Ejecutar main_app_final.py para correr pipeline de RAG completo
6) Para correr aplicación streamlit ejecutar en consola la sentencia: streamlit run main_app_final.py
7) En la app se pueden realizar preguntas acerca de la legislación de alimentos en idiomas español, inglés o portugues


# Resultado esperado

La app puede recibir preguntas como: "Quais são as normas para lactantes no Brasil?", "¿Cual es la norma más importante para el rotulado de alimentos?". La aplicación procesará a traves de un proceso RAG la solicitud y luego de unos segundo entregará una respuesta enriquecida con inteligencia artificial y los documentos más relevantes asociados. En la imagen a continuación se despliega un ejemplo de ello:

<img width="643" alt="image" src="https://github.com/user-attachments/assets/43cdfa69-2a28-4d9e-b9d6-7a72fcd83caa" />

# Benchmark - Analisis de resultados 

<img width="1162" alt="image" src="https://github.com/user-attachments/assets/79307506-dc87-4882-a556-ddca1a71ffc7" />


