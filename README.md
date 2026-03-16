# Advanced RAG Assistant

An **Advanced Retrieval-Augmented Generation (RAG) system** built using Python and LangChain.

This project demonstrates how to build the **core architecture of a GenAI document assistant** by combining document ingestion, text chunking, embeddings, and vector search.

The goal of this project is to build a **production-style RAG pipeline** that can integrate with local LLMs such as **Mistral**, **LLaMA**, and **Gemma**.

---

# Overview

Large Language Models cannot efficiently process large documents directly.

**Retrieval-Augmented Generation (RAG)** solves this problem by retrieving relevant information from documents and providing that context to an LLM before generating an answer.

This project demonstrates the **foundational architecture behind modern AI knowledge assistants.**

---

# Architecture

The system follows a standard Retrieval-Augmented Generation pipeline.

```
PDF Document
      │
      ▼
LangChain PDF Loader
      │
      ▼
Recursive Character Text Splitter
      │
      ▼
Text Chunks
      │
      ▼
Embedding Model (all-MiniLM-L6-v2)
      │
      ▼
FAISS Vector Database
      │
      ▼
Semantic Retrieval
      │
      ▼
LLM (Mistral / GPT)
      │
      ▼
Final AI Response
```

---

# Project Structure

```
advanced-rag-assistant
│
├── data
│   └── sample.pdf
│
├── src
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   └── vector_store.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Tech Stack

## Programming Language

Python

## AI / GenAI Frameworks

* LangChain
* Sentence Transformers
* FAISS Vector Database

## Concepts Implemented

* Document ingestion
* Text chunking
* Vector embeddings
* Semantic search
* Retrieval-Augmented Generation (RAG)

## Local LLM Support

This project is designed to integrate with local models such as:

* Mistral
* LLaMA
* Gemma

---

# Current Capabilities

The system currently implements the **core stages of a Retrieval-Augmented Generation pipeline.**

### Implemented Pipeline

```
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
↓
Similarity Retrieval
↓
LLM Answer Generation
```

---

# How the System Works

1. The system loads a PDF document using LangChain's `PyPDFLoader`.
2. The document is split into smaller semantic chunks using `RecursiveCharacterTextSplitter`.
3. Each chunk is converted into embeddings using the **Sentence Transformers model**.
4. The embeddings are stored inside a **FAISS vector database**.
5. When a user asks a question, the system retrieves the **most relevant document chunks**.
6. The retrieved context is passed to a **local LLM (Mistral)** to generate the final answer.

---

# Example Query

User Question

```
What is Retrieval-Augmented Generation?
```

System Process

1. Convert question into embedding
2. Search FAISS vector database
3. Retrieve most relevant document chunks
4. Send retrieved context to LLM
5. Generate final grounded response

Example Answer

```
Retrieval-Augmented Generation (RAG) is an architecture that combines
information retrieval with large language models. It retrieves relevant
information from external documents and uses that context to generate
accurate answers.
```

---

# Example Output

```
Loading document...
Splitting document into chunks...
Creating vector database...
Vector database created successfully
Total chunks stored: 16
```

---

# Metadata Extraction

Each text chunk retains metadata extracted from the original PDF.

Example metadata:

```json
{
  "source": "data/sample.pdf",
  "page": 1,
  "total_pages": 7,
  "author": "ashwin kumar m"
}
```

This metadata allows the system to provide **document citations for AI responses**.

---

# Setup Instructions

## Clone Repository

```
git clone https://github.com/Nithish7383/advanced-rag-assistant.git
cd advanced-rag-assistant
```

---

## Create Virtual Environment

```
python -m venv venv
```

Activate it (Windows)

```
venv\Scripts\activate
```

---

## Install Dependencies

```
pip install -r requirements.txt
```

---

## Run the Project

```
python main.py
```

---

# Development Progress

This repository is being built incrementally as part of a **GenAI engineering portfolio project**.

### Completed

Day 1

* Document ingestion
* Text chunking

Day 2

* Embedding generation
* FAISS vector database integration

Day 3

* Implemented semantic similarity search
* Query-based document retrieval
* Metadata-based source tracing

Day 4

* Integrated local Mistral LLM using Ollama
* Implemented answer generation from retrieved context
* Built full Retrieval-Augmented Generation pipeline

Day 5

* Retrieval-Augmented question answering

---


Day 6

* Chat interface for document interaction

# Upcoming Features


Future Improvements

* Hybrid search (BM25 + Vector Search)
* Multi-document ingestion
* Streaming responses
* RAG evaluation metrics
* Docker deployment
* Streamlit web interface

---

# Learning Goals

This project focuses on building **real-world GenAI engineering systems**, including:

* Retrieval-Augmented Generation
* AI knowledge assistants
* Semantic document search
* Local LLM applications
* AI-powered document understanding

---

# Author

**Nithish**

GenAI Engineer
LLM Applications • RAG Systems • AI Automation

GitHub
https://github.com/Nithish7383
