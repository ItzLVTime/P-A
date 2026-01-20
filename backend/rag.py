from backend.embeddings import embed_texts, EmbeddingError
from backend.vector_store import search, VectorStoreError
import logging

logger = logging.getLogger(__name__)

class RAGError(Exception):
    """Raised when RAG operations fail."""
    pass

def build_context(chunks, index, question, top_k=3):
    """
    Build context from chunks relevant to the question.

    Args:
        chunks: List of text chunks with page info
        index: FAISS index
        question: User question
        top_k: Number of chunks to retrieve

    Returns:
        Tuple of (context_string, list_of_page_numbers)

    Raises:
        RAGError: If context building fails
    """
    if not chunks:
        logger.error("No chunks available for context building")
        raise RAGError("No chunks available")
    if not question or not question.strip():
        logger.error("Empty question provided")
        raise RAGError("Question cannot be empty")

    logger.info(f"Building context for question: '{question[:50]}...'")
    try:
        question_embedding = embed_texts([question])
        ids = search(index, question_embedding, top_k)
    except (EmbeddingError, VectorStoreError) as e:
        logger.error(f"Failed to search for relevant context: {e}")
        raise RAGError(f"Failed to search for relevant context: {e}")

    context_blocks = []
    pages = set()

    for i in ids:
        if 0 <= i < len(chunks):
            context_blocks.append(chunks[i]["text"])
            pages.add(chunks[i]["page"])

    if not context_blocks:
        logger.error("No relevant context found")
        raise RAGError("No relevant context found")

    context = "\n\n".join(context_blocks)
    logger.info(f"Built context from {len(context_blocks)} chunks (pages: {sorted(pages)})")
    return context, sorted(pages)
