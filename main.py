from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from src.retriever import search_documents


def main():

    print("Loading document...")

    docs = load_pdf("data/sample.pdf")


    print("Splitting document into chunks...")

    chunks = split_documents(docs)


    print("Creating vector database...")

    vector_store = create_vector_store(chunks)

    print("Vector database created successfully")

    print("Total chunks stored:", len(chunks))


    while True:

        query = input("\nEnter your question about the document (or type 'exit'): ")

        if query.lower() == "exit":
            print("Exiting search system.")
            break


        results = search_documents(vector_store, query)


        print("\nTop relevant results:\n")


        for i, doc in enumerate(results):

            print(f"Result {i+1}")
            print("-" * 40)

            print("Answer context:")
            print(doc.page_content[:400])

            print("\nSource:", doc.metadata["source"])
            print("Page:", doc.metadata["page"] + 1)

            print("\n")


if __name__ == "__main__":
    main()