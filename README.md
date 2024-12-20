## Readme: Pipeline RAG, equipo legislación de Brasil

Alumnos:

Nicolás Oscar Castro Bize
Carlos Alberto Zurita Olea
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
6.  for running streamlit app please execute on console
    ```bash
    streamlit run main_app_final.py
7. The app will open in a new tab. Then make questions about brazilian food legislation

## Arquitectura de la solución RAG implementada

<img width="772" alt="image" src="https://github.com/user-attachments/assets/f325bb5d-c37a-410b-a545-6111c6536ae7" />

# Consideraciones

1) Descargar  o clonar repositorio desde el main
2) Instalar requirements.txt con las librerias utilizadas
3) Guardar o asegurar tener archivo biblioteca_de_alimentos.pdf disponible en data/biblioteca-de-alimentos.pdf
4) Ejecutar main_app.py para correr pipeline de RAG completo
5) Para correr aplicación streamlit ejecutar en consola la sentencia: streamlit run main_app_v3_hybrid.py
6) En la app se pueden realizar preguntas acerca de la legislación de alimentos en idiomas español, inglés o portugues


# Resultado esperado
En la ejecución este pipeline recibe una pregunta relacionada como: "Quais são as normas para lactantes no Brasil?", al ejecutar main_app.py el resultado es de la forma:

Result 1:
page_content='Lei 11.265/2006 – Regulamenta a comercialização de alimentos para lactentes e crianças de primeira infância 
e a de produtos de puericultura correlatos. 
 Alterada por: 
Lei 11.474/2007  – Altera a Lei no 10.188, de 12 de fevereiro de 2001, que cria o Programa de 
Arrendamento Residencial, institui o arrendamento residencial com opção de compra, e a Lei no 
11.265, de 3 de janeiro de 2006, que regulamenta a comercialização de alimentos para lactentes e 
crianças de primeira infância e a de produtos de puericultura correlatos, e dá outras providências. 

# Benchmark - Analisis de resultados 

<img width="1162" alt="image" src="https://github.com/user-attachments/assets/79307506-dc87-4882-a556-ddca1a71ffc7" />


***Readme anterior*******
Building an End-to-End Retrieval-Augmented Generation System

Welcome to the **Building an End-to-End Retrieval-Augmented Generation System** repository. This repository is designed to guide you through the process of creating a complete Retrieval-Augmented Generation (RAG) system from scratch, following a structured curriculum.

## Setup Instructions

To get started with the course:

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
## Table of Contents

- [Building an End-to-End Retrieval-Augmented Generation System](#building-an-end-to-end-retrieval-augmented-generation-system)
  - [Setup Instructions](#setup-instructions)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Course Outline](#course-outline)
    - [Lesson 1: Introduction to Retrieval-Augmented Generation (RAG)](#lesson-1-introduction-to-retrieval-augmented-generation-rag)
    - [Lesson 2: Document Chunking Strategies](#lesson-2-document-chunking-strategies)

## Introduction

This repository contains the materials and code needed to build a complete Retrieval-Augmented Generation (RAG) system. A RAG system combines the strengths of large language models with an external knowledge base to improve the accuracy and relevance of generated responses. Throughout this course, you'll gain hands-on experience with the various components of a RAG system, from document chunking to deployment in the cloud.

## Course Outline

### Lesson 1: Introduction to Retrieval-Augmented Generation (RAG)
- **Objective:** Understand the fundamentals of RAG and its applications.
- **Topics:**
  - Overview of RAG systems
  - Challenges in large language models (e.g., hallucinations, outdated information)
  - Basic components of a RAG system
- **Practical Task:** Set up your development environment and familiarize yourself with the basic concepts.
- **Resources:** 
  - Basics
  - More concepts

### Lesson 2: Document Chunking Strategies
- **Objective:** Learn how to effectively segment documents for better retrieval performance.
- **Topics:**
  - Chunking techniques: token-level, sentence-level, semantic-level
  - Balancing context preservation with retrieval precision
  - Small2Big and sliding window techniques
- **Practical Task:** Implement chunking strategies on a sample dataset.
- **Resources:**
  - The five levels of chunking
  - A guide to chunking

