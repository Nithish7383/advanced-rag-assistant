import streamlit as st
import time
import os
import shutil

from src.vector_store import load_vector_store, create_vector_store
from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker
from src.llm import stream_answer, extract_sources
from src.summarizer import generate_summary


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Advanced RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================
# CSS FIX
# =========================

st.markdown("""
<style>
button[data-testid="baseButton-secondary"] {
    padding: 4px 6px;
    height: 32px;
    width: 34px;
    font-size: 14px;
}

button[kind="secondary"] {
    width: auto !important;
}
</style>
""", unsafe_allow_html=True)


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("📄 Document AI Assistant")

    st.markdown("### Upload Document")

    uploaded_file = st.file_uploader("", type="pdf")

    st.markdown("---")

    st.subheader("📂 Knowledge Base")

    os.makedirs("uploaded_docs", exist_ok=True)

    files = os.listdir("uploaded_docs")

    st.caption(f"{len(files)} document(s) loaded")

    if files:

        for file in files:

            col1, col2 = st.columns([9,1])

            with col1:
                st.markdown(f"📄 **{file}**")

            with col2:
                if st.button("🗑", key=f"delete_{file}"):

                    os.remove(os.path.join("uploaded_docs", file))
                    st.rerun()

            st.markdown("---")

    else:
        st.write("No documents uploaded.")

    st.markdown("---")

    if st.button("🔄 Reset Knowledge Base", use_container_width=True):

        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")

        if os.path.exists("uploaded_docs"):
            shutil.rmtree("uploaded_docs")

        st.success("Knowledge base cleared")
        st.rerun()


# =========================
# MAIN TITLE
# =========================

st.title("💬 Advanced RAG Assistant")

st.write("Ask questions about your uploaded documents.")


# =========================
# DOCUMENT PROCESSING
# =========================

vector_store = None
documents = []
summary = None

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

        summary = generate_summary(chunks)

    st.success("Document processed successfully!")

else:

    with st.spinner("Loading knowledge base..."):

        vector_store = load_vector_store()

        if vector_store:
            documents = vector_store.similarity_search("test", k=50)
        else:
            documents = []


# =========================
# RETRIEVER + RERANKER
# =========================

retriever = HybridRetriever(vector_store, documents)

reranker = Reranker()


# =========================
# SESSION MEMORY
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []


# =========================
# DOCUMENT INSIGHTS
# =========================

if summary:

    with st.expander("📄 Document Insights", expanded=True):
        st.markdown(summary)


# =========================
# CHAT HISTORY
# =========================

chat_container = st.container()

with chat_container:

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# =========================
# CHAT INPUT
# =========================

query = st.chat_input("Ask anything about your documents...")


# =========================
# CHAT PROCESSING
# =========================

if query:

    with chat_container:

        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("assistant"):

            placeholder = st.empty()

            placeholder.markdown("🤖 Generating answer...")

            start_time = time.time()

            retrieved_docs = retriever.retrieve(query, k=10)

            docs = reranker.rerank(query, retrieved_docs, top_k=3)

            response = ""

            for token in stream_answer(query, docs, st.session_state.history):

                response += token
                placeholder.markdown(response + "▌")

            placeholder.markdown(response)

            end_time = time.time()

            response_time = round(end_time - start_time, 2)

            token_count = len(response.split())

            tokens_per_sec = round(token_count / response_time, 2) if response_time > 0 else 0


            st.session_state.history.append(f"User: {query}")
            st.session_state.history.append(f"Assistant: {response}")


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


# =========================
# CONTEXT + DEBUG PANELS
# =========================

if query:

    with st.expander("📚 Context Sent to LLM"):

        for i, doc in enumerate(docs):

            st.markdown(f"**Chunk {i+1}**")
            st.markdown(doc.page_content)
            st.markdown("---")


    with st.expander("🔎 Retrieval Debug Panel"):

        st.subheader("Vector Search Results")

        for doc in retriever.last_vector_results:
            st.markdown(doc.page_content[:300])
            st.markdown("---")

        st.subheader("BM25 Keyword Results")

        for doc in retriever.last_bm25_results:
            st.markdown(doc.page_content[:300])
            st.markdown("---")

        st.subheader("Final Hybrid Results")

        for doc in retriever.last_combined_results:
            st.markdown(doc.page_content[:300])
            st.markdown("---")


# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown(
    "<center style='color:gray'>Advanced RAG Assistant • Built with Streamlit, LangChain, FAISS, Sentence Transformers, and Ollama</center>",
    unsafe_allow_html=True
)