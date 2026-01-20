import streamlit as st
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from backend.pdf_loader import load_pdf, PDFLoadError
from backend.chunker import chunk_pages
from backend.embeddings import embed_texts, EmbeddingError
from backend.vector_store import build_index, VectorStoreError
from backend.qa_engine import answer_question, QAError
from backend.cache import get_file_hash, load_cached_index, save_to_cache, get_cache_stats, clear_expired_cache

st.set_page_config(page_title="P&A - PDF Q&A", layout="wide")
st.title("P&A - Ask Questions from Your PDF")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of context chunks", min_value=1, max_value=10, value=3)

    # Cache management buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear All"):
            from backend.cache import clear_cache
            clear_cache()
            st.success("Cache cleared!")
    with col2:
        if st.button("Clear Expired"):
            removed = clear_expired_cache()
            if removed > 0:
                st.success(f"Removed {removed} expired files")
            else:
                st.info("No expired files")

    # Show cache statistics
    stats = get_cache_stats()
    if stats["total_files"] > 0:
        st.caption(f"Cache: {stats['valid_files']} valid, {stats['expired_files']} expired ({stats['total_size_mb']} MB)")
        st.caption(f"TTL: {stats['ttl_days']} days")

    if "ready" in st.session_state:
        st.divider()
        st.caption(f"Chunks: {len(st.session_state.chunks)}")
        if "file_hash" in st.session_state:
            st.caption(f"File hash: {st.session_state.file_hash[:8]}...")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Check if this is a new file
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Reset for potential re-read

    # Compute hash of uploaded content
    import hashlib
    current_hash = hashlib.md5(file_bytes).hexdigest()

    # Only process if it's a new file
    if "file_hash" not in st.session_state or st.session_state.file_hash != current_hash:
        # Check cache first
        cached_chunks, cached_index = load_cached_index(current_hash)

        if cached_chunks is not None and cached_index is not None:
            st.session_state.chunks = cached_chunks
            st.session_state.index = cached_index
            st.session_state.file_hash = current_hash
            st.session_state.ready = True
            logger.info(f"PDF loaded from cache (hash: {current_hash[:8]}...)")
            st.success("PDF loaded from cache!")
        else:
            with st.spinner("Processing PDF..."):
                try:
                    os.makedirs("data", exist_ok=True)
                    pdf_path = "data/uploaded.pdf"

                    with open(pdf_path, "wb") as f:
                        f.write(file_bytes)

                    pages = load_pdf(pdf_path)
                    chunks = chunk_pages(pages)

                    texts = [c["text"] for c in chunks]
                    embeddings = embed_texts(texts)
                    index = build_index(embeddings)

                    # Save to cache
                    save_to_cache(current_hash, chunks, index)

                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.file_hash = current_hash
                    st.session_state.ready = True

                    logger.info(f"PDF processed: {len(chunks)} chunks, hash: {current_hash[:8]}...")
                    st.success(f"PDF processed successfully! ({len(chunks)} chunks)")

                except PDFLoadError as e:
                    st.error(f"Failed to load PDF: {e}")
                except EmbeddingError as e:
                    st.error(f"Failed to generate embeddings: {e}")
                except VectorStoreError as e:
                    st.error(f"Failed to build search index: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

if "ready" in st.session_state and st.session_state.ready:
    question = st.text_input("Ask a question from the PDF")

    if question:
        with st.spinner("Thinking..."):
            try:
                answer, pages = answer_question(
                    question,
                    st.session_state.chunks,
                    st.session_state.index,
                    top_k=top_k
                )

                st.subheader("Answer")
                st.write(answer)

                if pages:
                    st.caption(f"Source pages: {', '.join(map(str, pages))}")

            except QAError as e:
                st.error(f"Failed to answer question: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
