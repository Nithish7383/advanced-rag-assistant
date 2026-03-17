from src.llm import stream_answer

def generate_summary(docs):

    context = docs[:3]   # Only first chunks

    prompt = """
Summarize this document in 3 sentences.

Then list 4 key topics only.
"""

    response = ""

    for token in stream_answer(prompt, context, []):
        response += token

    return response