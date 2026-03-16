import streamlit as st
import time
from src.vector_store import load_vector_store
from src.llm import generate_answer

# Page configuration
st.set_page_config(page_title="Advanced RAG Assistant", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📄 Document AI Assistant")
    st.write("Built with LangChain + FAISS + Phi-3 Mini")

# Title
st.title("💬 Advanced RAG Assistant")

# Load vector store
with st.spinner("Loading knowledge base..."):
    vector_store = load_vector_store()

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
query = st.chat_input("Ask a question about your document")

if query:

    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("assistant"):

        placeholder = st.empty()

        placeholder.markdown("🤖 Thinking...")

        # Start timer
        start_time = time.time()

        # Retrieve documents
        docs = vector_store.similarity_search(query, k=2)

        # Generate answer
        response, sources = generate_answer(query, docs)

        # End timer
        end_time = time.time()

        response_time = round(end_time - start_time, 2)

        # Typing animation
        full_response = ""
        for word in response.split():
            full_response += word + " "
            time.sleep(0.02)
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

        # Token estimation
        token_count = len(response.split())
        tokens_per_sec = round(token_count / response_time, 2) if response_time > 0 else 0

        # Show sources
        if sources:
            st.markdown("**Sources:**")
            for page in sources:
                st.markdown(f"- Page {page}")

        # Performance metrics
        st.markdown("---")
        st.markdown(f"⏱️ **Response Time:** {response_time} seconds")
        st.markdown(f"🔢 **Tokens Generated:** {token_count}")
        st.markdown(f"⚡ **Tokens/sec:** {tokens_per_sec}")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )