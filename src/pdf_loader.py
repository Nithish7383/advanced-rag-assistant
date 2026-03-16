from langchain_community.document_loaders import PyPDFLoader
import os


def load_pdf(file_path):

    loader = PyPDFLoader(file_path)

    docs = loader.load()

    # Extract clean file name
    filename = os.path.basename(file_path)

    # Store filename in metadata
    for doc in docs:
        doc.metadata["source"] = filename

    return docs