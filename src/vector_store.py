from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import streamlit as st


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_store(chunks):

    embeddings = get_embeddings()

    # If FAISS already exists, load and append
    if os.path.exists("faiss_index"):

        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        vector_store.add_documents(chunks)

    else:

        vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local("faiss_index")

    return vector_store


@st.cache_resource
def load_vector_store():

    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store