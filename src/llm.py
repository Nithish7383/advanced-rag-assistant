from langchain_community.llms import Ollama


# Load the local LLM (Phi-3 Mini)
def load_llm():

    llm = Ollama(
        model="phi3:mini"
    )

    return llm


# Generate answer using retrieved context
def generate_answer(query, docs):

    llm = load_llm()

    # If no documents found
    if not docs:
        return "I could not find this information in the document."

    # Combine retrieved chunks
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a document question-answering assistant.

Use ONLY the provided context to answer the question.

If the answer cannot be found in the context, respond with:
"I could not find this information in the document."

Do NOT use outside knowledge.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response