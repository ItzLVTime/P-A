import re
import logging

logger = logging.getLogger(__name__)


def split_into_sentences(text):
    """
    Split text into sentences.
    Handles common sentence endings: . ! ? and also handles abbreviations better.
    """
    # Split on sentence-ending punctuation followed by space and capital letter
    # This pattern keeps the punctuation with the sentence
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)

    # Clean up and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def chunk_pages(pages, chunk_size=500, overlap=100):
    """
    Split pages into chunks using sentence-aware splitting.

    Args:
        pages: List of dicts with 'page' and 'text' keys
        chunk_size: Target chunk size in words (soft limit)
        overlap: Number of words to overlap between chunks

    Returns:
        List of dicts with 'page' and 'text' keys
    """
    chunks = []

    for page in pages:
        text = page["text"]
        page_num = page["page"]

        # Split into sentences
        sentences = split_into_sentences(text)

        if not sentences:
            continue

        current_chunk_sentences = []
        current_word_count = 0

        for sentence in sentences:
            sentence_word_count = len(sentence.split())

            # If adding this sentence exceeds chunk_size and we have content
            if current_word_count + sentence_word_count > chunk_size and current_chunk_sentences:
                # Save current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append({
                    "page": page_num,
                    "text": chunk_text
                })

                # Start new chunk with overlap
                # Keep last few sentences for overlap
                overlap_sentences = []
                overlap_words = 0

                for s in reversed(current_chunk_sentences):
                    s_words = len(s.split())
                    if overlap_words + s_words <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_words += s_words
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_word_count = overlap_words

            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_word_count

        # Don't forget the last chunk of the page
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "page": page_num,
                "text": chunk_text
            })

    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages (sentence-aware)")
    return chunks
