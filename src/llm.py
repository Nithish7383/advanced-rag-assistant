import ollama


def generate_answer(query, context_chunks):
    """
    Generate an answer using the Mistral model
    based on retrieved document chunks.
    """

    context = "\n\n".join([doc.page_content for doc in context_chunks])

    prompt = f"""
You are a helpful assistant answering questions using the provided document context.

Context:
{context}

Question:
{query}

Answer clearly using the information from the context.
"""

    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]