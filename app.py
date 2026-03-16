import streamlit as st
import time
from src.vector_store import load_vector_store
from langchain_community.llms import Ollama

# Page configuration
st.set_page_config(page_title="Advanced RAG Assistant", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📄 Document AI Assistant")
    st.write("Built with LangChain + FAISS + Mistral")

# Title
st.title("💬 Advanced RAG Assistant")

# Load vector store
with st.spinner("Loading knowledge base..."):
    vector_store = load_vector_store()

# Load LLM
llm = Ollama(model="mistral")

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
query = st.chat_input("Ask a question about your document")

if query:

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    # Assistant thinking indicator
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("🤖 Thinking..."):

            # Retrieve docs
            docs = vector_store.similarity_search(query, k=3)

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
You are an AI assistant answering questions using the document context.

Context:
{context}

Question:
{query}

Answer:
"""

            response = llm.invoke(prompt)

        # Streaming typing effect
        full_response = ""
        for word in response.split():
            full_response += word + " "
            time.sleep(0.03)
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )