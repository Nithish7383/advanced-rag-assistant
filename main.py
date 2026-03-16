from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.vector_store import create_vector_store


print("Loading document...")

docs = load_pdf("data/sample.pdf")

print("Splitting document into chunks...")

chunks = split_documents(docs)

print("Creating vector database...")

vector_store = create_vector_store(chunks)

print("Vector database created successfully")

print("Total chunks stored:", len(chunks))

print(chunks[4].metadata)