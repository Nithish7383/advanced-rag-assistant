import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vector_store(chunks):

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    os.makedirs("faiss_index", exist_ok=True)

    vector_store.save_local("faiss_index")

    return vector_store


def load_vector_store():

    index_path = "faiss_index"

    # if index does not exist yet
    if not os.path.exists(index_path):
        return None

    embeddings = get_embeddings()

    try:

        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vector_store

    except Exception:
        return None