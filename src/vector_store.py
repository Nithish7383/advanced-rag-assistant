from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Embedding model
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Create and save FAISS vector store
def create_vector_store(chunks):

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(chunks, embeddings)

    # save index locally
    vector_store.save_local("faiss_index")

    print("Vector database created successfully")

    return vector_store


# Load existing FAISS vector store
def load_vector_store():

    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store