import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

AVAILABLE_MODELS = {
    "Llama 3.3 70B": "llama-3.3-70b-versatile",
    "Llama 3.1 8B": "llama-3.1-8b-instant",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Gemma 2 9B": "gemma2-9b-it"
}

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def load_llm(model_name=DEFAULT_MODEL):
    return model_name


def stream_answer(query, docs, history, model_name=DEFAULT_MODEL):

    if not docs:
        yield "I could not find this information in the document."
        return

    context = "\n\n".join([doc.page_content for doc in docs])
    conversation = "\n".join(history)

    prompt = f"""You are a document question answering system.

Use the conversation history and the context to answer the question.

Answer ONLY using the provided context.

If the answer is not present in the context say:
"I could not find this information in the document."

Conversation History:
{conversation}

Context:
{context}

Question:
{query}

Answer:"""

    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0.2,
        max_tokens=1024
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token


def extract_sources(docs):

    sources = []

    for doc in docs:
        file_name = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "Unknown Page")
        sources.append((file_name, page))

    return list(set(sources))