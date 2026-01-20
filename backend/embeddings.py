from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import logging
import os

load_dotenv()

logger = logging.getLogger(__name__)

# Default embedding model - can be overridden via EMBEDDING_MODEL env var
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass

_model = None
_model_name = None

def get_model():
    """Lazy load the embedding model."""
    global _model, _model_name

    if _model is None:
        # Get model name from environment or use default
        _model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        logger.info(f"Loading embedding model: {_model_name}")
        try:
            _model = SentenceTransformer(_model_name)
            logger.info(f"Embedding model '{_model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{_model_name}': {e}")
            raise EmbeddingError(f"Failed to load embedding model '{_model_name}': {e}")
    return _model


def get_model_name():
    """Get the name of the currently loaded model."""
    global _model_name
    if _model_name is None:
        _model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return _model_name

def embed_texts(texts):
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of strings to embed

    Returns:
        numpy array of embeddings

    Raises:
        EmbeddingError: If embedding generation fails
    """
    if not texts:
        logger.error("No texts provided for embedding")
        raise EmbeddingError("No texts provided for embedding")

    logger.debug(f"Generating embeddings for {len(texts)} texts")
    try:
        model = get_model()
        embeddings = model.encode(texts, show_progress_bar=False)
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    except EmbeddingError:
        raise
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise EmbeddingError(f"Failed to generate embeddings: {e}")
