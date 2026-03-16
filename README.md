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


Document (PDF)
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
Embedding Model
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
│ └── sample.pdf # Example document
│
├── src
│ ├── pdf_loader.py # PDF document loader
│ └── text_splitter.py # Text chunking logic
│
├── main.py # Entry point of the application
├── requirements.txt # Project dependencies
└── README.md # Project documentation


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

# Features Implemented

- PDF document loading
- Recursive text chunking
- Modular code architecture
- Clean project structure for GenAI systems
- Foundation for vector search

---

# Setup Instructions

## Clone Repository


git clone https://github.com/Nithish7383/advanced-rag-assistant.git

cd advanced-rag-assistant


---

## Create Virtual Environment


python -m venv venv


Activate it:

Windows


venv\Scripts\activate


---

## Install Dependencies


pip install -r requirements.txt


---

## Run the Project


python main.py


Example Output


Total chunks: 16

Enterprise Email Automation
Proposal — Outlook
Limitation, Proof, and Azure
Logic Apps Solution


---

# Current Pipeline

The current system performs the following steps:


PDF Document
↓
Load document using LangChain
↓
Split document into semantic chunks
↓
Prepare chunks for embedding generation


---

# Next Development Steps

Planned improvements:

- Add embedding generation
- Implement FAISS vector database
- Add semantic retrieval
- Connect to local LLM (Mistral)
- Implement question-answering interface
- Add document citation support
- Build chat-based UI

---

# Learning Objectives

This project is part of a **GenAI engineering portfolio**, focusing on building real-world AI systems such as:

- Retrieval-Augmented Generation (RAG)
- AI knowledge assistants
- Document search systems
- Local LLM applications

---

# Author

**Nithish**

GenAI Engineer  
AI Systems • LLM Applications • RAG Pipelines

GitHub  
https://github.com/Nithish7383