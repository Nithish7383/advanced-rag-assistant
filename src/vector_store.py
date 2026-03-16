import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


INDEX_PATH = "faiss_index"


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_or_load_vector_store(chunks):

    embeddings = get_embedding_model()

    # If FAISS index already exists → load it
    if os.path.exists(INDEX_PATH):

        print("Loading existing FAISS index...")

        vector_store = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vector_store

    # Otherwise create new index
    print("Creating new FAISS index...")

    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(INDEX_PATH)

    print("FAISS index saved locally.")

    return vector_store