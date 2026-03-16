from langchain_community.llms import Ollama


def load_llm():

    llm = Ollama(model="phi3:mini")

    return llm


def generate_answer(query, docs):

    llm = load_llm()

    if not docs:
        return "I could not find this information in the document.", []

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a document question answering system.

Answer ONLY using the information provided in the context below.

If the answer is not present in the context, respond exactly with:

"I could not find this information in the document."

Do NOT use any external knowledge.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    sources = []

    for doc in docs:

        file_name = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "Unknown Page")

        sources.append((file_name, page))

    sources = list(set(sources))

    return response, sources