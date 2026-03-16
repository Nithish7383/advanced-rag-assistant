# Advanced RAG Assistant

A **Retrieval-Augmented Generation (RAG)** system built using Python and LangChain.

This project demonstrates how to build the **core architecture of a GenAI document assistant** by combining document ingestion, text chunking, embeddings, and vector search.

The goal of this project is to build a **production-style RAG pipeline that can later integrate with local LLMs such as Mistral.**

---

# Overview

Large Language Models cannot directly read large documents efficiently.

RAG (Retrieval-Augmented Generation) solves this problem by:

1. Breaking documents into smaller chunks
2. Converting them into vector embeddings
3. Storing them in a vector database
4. Retrieving relevant chunks during question answering

This project implements the **first stages of a scalable RAG system.**

---

# Architecture


PDF Document
│
▼
LangChain PDF Loader
│
▼
Recursive Text Splitter
│
▼
Text Chunks
│
▼
Embedding Model (Sentence Transformers)
│
▼
Vector Database (FAISS)
│
▼
Semantic Retrieval
│
▼
LLM (Mistral / GPT / etc.)
│
▼
Final AI Response


---

# Project Structure


advanced-rag-assistant
│
├── data
│ └── sample.pdf
│
├── src
│ ├── pdf_loader.py
│ ├── text_splitter.py
│ └── vector_store.py
│
├── main.py
├── requirements.txt
└── README.md


---

# Tech Stack

**Programming Language**

Python

**AI / GenAI Frameworks**

- LangChain
- Sentence Transformers
- FAISS Vector Database

**Concepts Implemented**

- Document ingestion
- Text chunking
- RAG pipeline foundation
- Vector database preparation

**Local LLM Support**

This project is designed to integrate with local models such as:

- Mistral
- LLaMA
- Gemma

---

# Current Capabilities

The system currently implements the **core stages of a Retrieval-Augmented Generation (RAG) pipeline**.

### Implemented Pipeline


PDF Document
↓
LangChain PDF Loader
↓
Recursive Text Splitter
↓
Semantic Text Chunks
↓
Embedding Generation
↓
Vector Database (FAISS)


### What Happens Internally

1. The system loads a PDF document using LangChain's `PyPDFLoader`.
2. The document is split into smaller semantic chunks using `RecursiveCharacterTextSplitter`.
3. Each chunk is converted into vector embeddings using the `all-MiniLM-L6-v2` embedding model.
4. The embeddings are stored in a **FAISS vector database** for semantic similarity search.

### Example Output


Loading document...
Splitting document into chunks...
Creating vector database...
Vector database created successfully
Total chunks stored: 16


---

# Metadata Extraction

Each chunk of text retains metadata extracted from the original PDF document.

Example metadata:


{
'source': 'data/sample.pdf',
'page': 1,
'total_pages': 7,
'author': 'ashwin kumar m'
}


This metadata is used later to provide **document citations in AI responses**.

---

Day-3

- Implemented semantic similarity search
- Added query-based document retrieval
- Integrated FAISS similarity search
- Enabled metadata-based source tracing

# Setup Instructions

## Clone Repository


git clone https://github.com/Nithish7383/advanced-rag-assistant.git

cd advanced-rag-assistant


---

## Create Virtual Environment


python -m venv venv


Activate it (Windows):


venv\Scripts\activate


---

## Install Dependencies


pip install -r requirements.txt


---

## Run the Project


python main.py


---

# Development Progress

This repository is being built incrementally as part of a **GenAI engineering portfolio project**.

### Completed

Day 1
- Document ingestion
- Text chunking

Day 2
- Embedding generation
- FAISS vector database integration

### Upcoming

Day 3
- Semantic similarity search

Day 4
- LLM integration (Mistral)

Day 5
- Retrieval-Augmented question answering

Day 6
- Chat interface for document interaction

---

# Learning Goals

This project focuses on building real-world **GenAI engineering systems** such as:

- Retrieval-Augmented Generation (RAG)
- AI knowledge assistants
- Semantic document search
- Local LLM applications

---

# Author

**Nithish**

GenAI Engineer  
LLM Applications • RAG Systems • AI Automation

GitHub  
https://github.com/Nithish7383