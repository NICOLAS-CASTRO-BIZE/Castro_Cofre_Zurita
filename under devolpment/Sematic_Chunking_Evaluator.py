
from langchain_community.document_loaders import PyPDFLoader

# Paso 1: Definir la ruta del archivo PDF

# Ruta del archivo PDF (asegúrate de que el archivo esté en la misma carpeta o usa una ruta completa)
ruta_pdf = "../data/biblioteca-de-alimentos2.pdf"

# Paso 2: Cargar el archivo PDF con PyPDFLoader
loader = PyPDFLoader(ruta_pdf)

# Extraer las páginas del PDF
documento = loader.load()

# Concatenar el texto de todas las páginas en un solo string
texto_pdf = "\n".join([page.page_content for page in documento])


#Paso 3: Exportar el texto extraído a un archivo .txt
ruta_txt = "archivo_extraido.txt"
with open(ruta_txt, "w", encoding="utf-8") as archivo_txt:
    archivo_txt.write(texto_pdf)

# Confirmación
print(f"El texto extraído del PDF se ha guardado en el archivo: {ruta_txt}")

# ## Recursive Character Splitter


from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 215, chunk_overlap=0)
text_splitter.create_documents([texto_pdf])

# ## Guardado de los Chunks

# import os

# # Create the new folder if it doesn't exist
# os.makedirs("Chunks_1", exist_ok=True)

# # Split the text into chunks
# chunks = text_splitter.split_text(texto_pdf)

# # Save each chunk into an individual .txt file
# for i, chunk in enumerate(chunks):
#     chunk_filename = f"Chunks_1/chunk_{i+1}.txt"
#     with open(chunk_filename, "w", encoding="utf-8") as chunk_file:
#         chunk_file.write(chunk)

# print(f"Saved {len(chunks)} chunks into the 'Chunks_1' folder.")


# ## Semantic Chunking

# ### División de texto en oraciones

import re

# Splitting the essay on '.', '?', and '!'
single_sentences_list = re.split(r'(?<=[.?!])\s+', texto_pdf)
print (f"{len(single_sentences_list)} sentences were found")


sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(single_sentences_list)]
#sentences[:10]



def combine_sentences(sentences, buffer_size=1):
    # Go through each sentence dict
    for i in range(len(sentences)):

        # Create a string that will hold the sentences which are joined
        combined_sentence = ''

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]['sentence'] + ' '

        # Add the current sentence
        combined_sentence += sentences[i]['sentence']

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += ' ' + sentences[j]['sentence']

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]['combined_sentence'] = combined_sentence

    return sentences

sentences = combine_sentences(sentences)



from langchain.embeddings import OpenAIEmbeddings
oaiembeds = OpenAIEmbeddings()

# Get the embeddings for the combined sentences
embeddings = oaiembeds.embed_documents([x['combined_sentence'] for x in sentences])

# Store the embeddings in the sentence dictionaries

for i, sentence in enumerate(sentences):
    sentence['combined_sentence_embedding'] = embeddings[i]


from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        
        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]['distance_to_next'] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences


distances, sentences = calculate_cosine_distances(sentences)



# We need to get the distance threshold that we'll consider an outlier
# We'll use numpy .percentile() for this

import numpy as np
breakpoint_percentile_threshold = 79

breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold) # If you want more chunks, lower the percentile cutoff


# Then we'll see how many distances are actually above this one
num_distances_above_theshold = len([x for x in distances if x > breakpoint_distance_threshold]) # The amount of distances above your threshold

# Then we'll get the index of the distances that are above the threshold. This will tell us where we should split our text
indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] # The indices of those breakpoints on your list

# Start of the shading and text
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for i, breakpoint_index in enumerate(indices_above_thresh):
    start_index = 0 if i == 0 else indices_above_thresh[i - 1]
    end_index = breakpoint_index if i < len(indices_above_thresh) - 1 else len(distances)

    

# # Additional step to shade from the last breakpoint to the end of the dataset
if indices_above_thresh:
    last_breakpoint = indices_above_thresh[-1]
    if last_breakpoint < len(distances):
        

# Initialize the start index
start_index = 0

# Create a list to hold the grouped sentences
chunks = []

# Iterate through the breakpoints to slice the sentences
for index in indices_above_thresh:
    # The end index is the current breakpoint
    end_index = index

    # Slice the sentence_dicts from the current start index to the end index
    group = sentences[start_index:end_index + 1]
    combined_text = ' '.join([d['sentence'] for d in group])
    chunks.append(combined_text)
    
    # Update the start index for the next group
    start_index = index + 1

# The last group, if any sentences remain
if start_index < len(sentences):
    combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
    chunks.append(combined_text)


# # ## Guardado de los Chunks

# import os

# # Create the new folder if it doesn't exist
# os.makedirs("Semantic_Chunks", exist_ok=True)

# # Split the text into chunks
# #chunks = text_splitter.split_text(texto_pdf)

# # Save each chunk into an individual .txt file
# for i, chunk in enumerate(chunks):
#     chunk_filename = f"Semantic_Chunks/chunk_{i+1}.txt"
#     with open(chunk_filename, "w", encoding="utf-8") as chunk_file:
#         chunk_file.write(chunk)

# print(f"Saved {len(chunks)} chunks into the 'Sematic_Chunks' folder.")

