import streamlit as st
import os

from backend.pdf_loader import load_pdf
from backend.chunker import chunk_pages
from backend.embeddings import embed_texts
from backend.vector_store import build_index
from backend.qa_engine import answer_question

st.set_page_config(page_title="P&A - PDF Q&A", layout="wide")
st.title("📄 P&A – Ask Questions from Your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    if "ready" not in st.session_state:
        with st.spinner("Processing PDF..."):
            os.makedirs("data", exist_ok=True)
            pdf_path = "data/uploaded.pdf"

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            pages = load_pdf(pdf_path)
            chunks = chunk_pages(pages)

            texts = [c["text"] for c in chunks]
            embeddings = embed_texts(texts)
            index = build_index(embeddings)

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.ready = True

        st.success("PDF processed successfully!")

if "ready" in st.session_state:
    question = st.text_input("Ask a question from the PDF")

    if question:
        with st.spinner("Thinking..."):
            answer, pages = answer_question(
                question,
                st.session_state.chunks,
                st.session_state.index
            )

        st.subheader("Answer")
        st.write(answer)

        if pages:
            st.caption(f"📌 Source pages: {', '.join(map(str, pages))}")
