import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VectorStoreError(Exception):
    """Raised when vector store operations fail."""
    pass

def build_index(embeddings):
    """
    Build a FAISS index from embeddings.

    Args:
        embeddings: numpy array of embeddings

    Returns:
        FAISS index

    Raises:
        VectorStoreError: If index building fails
    """
    if embeddings is None or len(embeddings) == 0:
        logger.error("No embeddings provided for index building")
        raise VectorStoreError("No embeddings provided")

    logger.info(f"Building FAISS index with {len(embeddings)} vectors")
    try:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        logger.info(f"FAISS index built successfully (dimension: {dim})")
        return index
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        raise VectorStoreError(f"Failed to build index: {e}")

def search(index, query_embedding, top_k=3):
    """
    Search the index for similar vectors.

    Args:
        index: FAISS index
        query_embedding: Query vector(s)
        top_k: Number of results to return

    Returns:
        Array of indices for top matches

    Raises:
        VectorStoreError: If search fails
    """
    if index is None:
        logger.error("Index not initialized")
        raise VectorStoreError("Index not initialized")

    logger.debug(f"Searching index for top {top_k} matches")
    try:
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        distances, indices = index.search(query_embedding, top_k)
        logger.debug(f"Search completed, found {len(indices[0])} results")
        return indices[0]
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise VectorStoreError(f"Search failed: {e}")
