from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyPDFLoader


# Modelo gratuito de Hugging Face
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Texto a fragmentar

# Leer texto del PDF


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


file_path = "data/biblioteca-de-alimentos.pdf"  # Cambia esto por la ruta a tu archivo PDF
text = load_pdf(file_path)


#text = """La inteligencia artificial está transformando industrias. 
#Las redes neuronales profundas son una tecnología clave en este cambio. 
#El procesamiento de lenguaje natural permite que las máquinas entiendan texto. 
#El aprendizaje automático mejora con grandes cantidades de datos."""

# Fragmentar por oraciones
sentences = text.split(". ")

# Crear embeddings para cada oración
def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

embeddings = [get_embedding(sentence) for sentence in sentences]

# Calcular similitudes
similarity_matrix = cosine_similarity(torch.stack(embeddings).numpy())

# Agrupar fragmentos por similitud
grouped_chunks = []
visited = set()

for i, sentence in enumerate(sentences):
    if i not in visited:
        group = [sentence]
        visited.add(i)
        for j, sim in enumerate(similarity_matrix[i]):
            if sim > 0.8 and j not in visited:  # Umbral de similitud
                group.append(sentences[j])
                visited.add(j)
        grouped_chunks.append(" ".join(group))

print(grouped_chunks)



