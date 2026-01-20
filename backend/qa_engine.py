from backend.rag import build_context, RAGError
from backend.llm import ask_llm, LLMError
from backend.sanitizer import sanitize_question, sanitize_for_prompt, SanitizationError
import logging

logger = logging.getLogger(__name__)

class QAError(Exception):
    """Raised when Q&A operations fail."""
    pass

def answer_question(question, chunks, index, top_k=3):
    """
    Answer a question using the document chunks.

    Args:
        question: User question
        chunks: List of text chunks with page info
        index: FAISS index
        top_k: Number of chunks to use for context

    Returns:
        Tuple of (answer_string, list_of_page_numbers)

    Raises:
        QAError: If answering fails
    """
    # Sanitize the question first
    try:
        question = sanitize_question(question)
    except SanitizationError as e:
        logger.warning(f"Question sanitization failed: {e}")
        raise QAError(str(e))

    logger.info(f"Answering question: '{question[:50]}...'")
    try:
        context, pages = build_context(chunks, index, question, top_k=top_k)
        logger.debug(f"Context built from pages: {pages}")
    except RAGError as e:
        logger.error(f"Failed to build context: {e}")
        raise QAError(f"Failed to build context: {e}")

    # Sanitize the context as well (removes control characters)
    safe_context = sanitize_for_prompt(context)

    prompt = f"""You are a document assistant.
Answer ONLY using the context below.
If the answer is not present, say "Not found in the document."

Context:
{safe_context}

Question:
{question}

Answer:"""

    try:
        answer = ask_llm(prompt)
        logger.info(f"Answer generated successfully from pages {pages}")
    except LLMError as e:
        logger.error(f"Failed to get answer from LLM: {e}")
        raise QAError(f"Failed to get answer from LLM: {e}")

    return answer.strip(), pages
