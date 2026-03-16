import streamlit as st
import time
import os

from src.vector_store import load_vector_store, create_vector_store
from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.llm import generate_answer

# Page configuration
st.set_page_config(page_title="Advanced RAG Assistant", layout="wide")

# Sidebar
with st.sidebar:
    st.title("📄 Document AI Assistant")
    st.write("Built with LangChain + FAISS + Phi-3 Mini")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Title
st.title("💬 Advanced RAG Assistant")

vector_store = None

# Handle uploaded PDF
if uploaded_file:

    os.makedirs("uploaded_docs", exist_ok=True)
    file_path = os.path.join("uploaded_docs", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document..."):

        documents = load_pdf(file_path)
        chunks = split_documents(documents)

        vector_store = create_vector_store(chunks)

    st.success("Document processed successfully!")

else:

    with st.spinner("Loading existing knowledge base..."):
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

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):

        placeholder = st.empty()
        placeholder.markdown("🤖 Thinking...")

        start_time = time.time()

        docs = vector_store.similarity_search(query, k=2)

        response, sources = generate_answer(query, docs)

        end_time = time.time()

        response_time = round(end_time - start_time, 2)

        full_response = ""
        for word in response.split():
            full_response += word + " "
            time.sleep(0.02)
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

        token_count = len(response.split())
        tokens_per_sec = round(token_count / response_time, 2) if response_time > 0 else 0

        if sources:
            st.markdown("**Sources:**")
            for page in sources:
                st.markdown(f"- Page {page}")

        st.markdown("---")
        st.markdown(f"⏱️ **Response Time:** {response_time} seconds")
        st.markdown(f"🔢 **Tokens Generated:** {token_count}")
        st.markdown(f"⚡ **Tokens/sec:** {tokens_per_sec}")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )