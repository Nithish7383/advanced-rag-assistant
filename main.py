from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.vector_store import create_or_load_vector_store
from src.retriever import search_documents
from src.llm import generate_answer


def main():

    print("Loading document...")
    docs = load_pdf("data/sample.pdf")

    print("Splitting document into chunks...")
    chunks = split_documents(docs)

    print("Preparing vector database...")

    vector_store = create_or_load_vector_store(chunks)

    print("Vector database ready")


    while True:

        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            print("Goodbye.")
            break


        print("\nSearching documents...\n")

        results = search_documents(vector_store, query)


        print("Generating answer using Mistral...\n")

        answer = generate_answer(query, results)

        print("AI Answer:\n")
        print(answer)


        print("\nSources:\n")

        for doc in results:
            print(
                f"Page {doc.metadata['page'] + 1} | {doc.metadata['source']}"
            )


if __name__ == "__main__":
    main()