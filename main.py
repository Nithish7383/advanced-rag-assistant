from src.pdf_loader import load_pdf
from src.text_splitter import split_documents

docs = load_pdf("data/sample.pdf")

chunks = split_documents(docs)

print("Total chunks:", len(chunks))
print("\nFirst chunk:\n")
print(chunks[0].page_content[:300])