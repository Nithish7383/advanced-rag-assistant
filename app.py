import streamlit as st
import time
import os
import shutil

from src.vector_store import load_vector_store, create_vector_store
from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.hybrid_retriever import HybridRetriever
from src.llm import stream_answer, extract_sources


st.set_page_config(page_title="Advanced RAG Assistant", layout="wide")


# Sidebar
with st.sidebar:

    st.title("📄 Document AI Assistant")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    st.markdown("---")
    st.subheader("📂 Uploaded Documents")

    os.makedirs("uploaded_docs", exist_ok=True)

    files = os.listdir("uploaded_docs")

    if files:

        for file in files:

            col1, col2 = st.columns([4,1])

            col1.write(file)

            if col2.button("❌", key=file):

                os.remove(os.path.join("uploaded_docs", file))

                st.success(f"{file} deleted")

                st.rerun()

    else:

        st.write("No documents uploaded.")

    st.markdown("---")

    if st.button("Reset Knowledge Base"):

        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")

        if os.path.exists("uploaded_docs"):
            shutil.rmtree("uploaded_docs")

        st.success("Knowledge base cleared")

        st.rerun()


# Main Title
st.title("💬 Advanced RAG Assistant")


vector_store = None
documents = []


# Handle uploaded PDF
if uploaded_file:

    os.makedirs("uploaded_docs", exist_ok=True)

    file_path = os.path.join("uploaded_docs", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document..."):

        docs = load_pdf(file_path)

        chunks = split_documents(docs)

        documents = chunks

        vector_store = create_vector_store(chunks)

    st.success("Document processed successfully!")


else:

    with st.spinner("Loading knowledge base..."):

        vector_store = load_vector_store()

        documents = vector_store.similarity_search("test", k=50)


# Hybrid Retriever
retriever = HybridRetriever(vector_store, documents)


# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []


# Display chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


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

        start_time = time.time()

        # Hybrid retrieval
        docs = retriever.retrieve(query, k=3)

        response = ""

        # Streaming response
        for token in stream_answer(query, docs, st.session_state.history):

            response += token

            placeholder.markdown(response + "▌")

        placeholder.markdown(response)

        end_time = time.time()

        response_time = round(end_time - start_time, 2)

        token_count = len(response.split())

        tokens_per_sec = round(token_count / response_time, 2) if response_time > 0 else 0


        # Save conversation history
        st.session_state.history.append(f"User: {query}")
        st.session_state.history.append(f"Assistant: {response}")


        # Extract sources
        sources = extract_sources(docs)

        if sources:

            st.markdown("**Sources:**")

            for file_name, page in sources:
                st.markdown(f"- {file_name} — Page {page}")


        st.markdown("---")

        st.markdown(f"⏱ **Response Time:** {response_time} seconds")

        st.markdown(f"🔢 **Tokens Generated:** {token_count}")

        st.markdown(f"⚡ **Tokens/sec:** {tokens_per_sec}")


    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )