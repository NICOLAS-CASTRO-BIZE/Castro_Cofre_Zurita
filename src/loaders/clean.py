import re

def clean_text(text: str) -> str:
    """
    Limpia el texto eliminando saltos de línea innecesarios y caracteres especiales.
    
    Args:
        text: Texto extraído del PDF.
    
    Returns:
        Texto limpio.
    """
    # Eliminar saltos de línea y reemplazar múltiples espacios
    text = re.sub(r'\s*\n\s*', ' ', text)
    # Unir palabras separadas con guiones al final de la línea
    text = re.sub(r'-\s', '', text)
    # Eliminar espacios adicionales
    text = re.sub(r'\s+', ' ', text).strip()
    return text
