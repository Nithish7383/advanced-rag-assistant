from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st


# Load embedding model
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Create vector store from document chunks
def create_vector_store(chunks):

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local("faiss_index")

    print("Vector database created successfully")

    return vector_store


# Load existing vector store
@st.cache_resource
def load_vector_store():

    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store