from langchain_community.llms import Ollama


# Load local LLM
def load_llm():
    llm = Ollama(model="phi3:mini")
    return llm


def generate_answer(query, docs):

    llm = load_llm()

    if not docs:
        return "I could not find this information in the document.", []

    # Combine context
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a document question-answering assistant.

Use ONLY the provided context to answer the question.

If the answer cannot be found in the context, say:
"I could not find this information in the document."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    # Extract source pages
    sources = list(set([doc.metadata.get("page", "Unknown") for doc in docs]))

    return response, sources