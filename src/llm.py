from langchain_community.llms import Ollama


def load_llm():
    llm = Ollama(model="phi3:mini")
    return llm


def stream_answer(query, docs, history):

    llm = load_llm()

    if not docs:
        yield "I could not find this information in the document."
        return

    context = "\n\n".join([doc.page_content for doc in docs])

    conversation = "\n".join(history)

    prompt = f"""
You are a document question answering system.

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

Answer:
"""

    for chunk in llm.stream(prompt):
        yield chunk


def extract_sources(docs):

    sources = []

    for doc in docs:

        file_name = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "Unknown Page")

        sources.append((file_name, page))

    return list(set(sources))