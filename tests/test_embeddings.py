import pytest
import numpy as np
from backend.embeddings import embed_texts, EmbeddingError


class TestEmbedTexts:
    def test_embed_single_text(self):
        """Test embedding a single text."""
        embeddings = embed_texts(["Hello world"])

        assert embeddings is not None
        assert len(embeddings) == 1
        assert embeddings.shape[1] == 384  # MiniLM-L6-v2 dimension

    def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = embed_texts(texts)

        assert len(embeddings) == 3
        assert embeddings.shape == (3, 384)

    def test_embeddings_are_normalized(self):
        """Test that embeddings have reasonable magnitude."""
        embeddings = embed_texts(["Test sentence"])
        norm = np.linalg.norm(embeddings[0])

        # MiniLM produces normalized embeddings (norm close to 1)
        assert 0.5 < norm < 2.0

    def test_similar_texts_have_similar_embeddings(self):
        """Test that similar texts produce similar embeddings."""
        texts = [
            "The cat sat on the mat",
            "A cat is sitting on a mat",
            "Quantum physics is complicated"
        ]
        embeddings = embed_texts(texts)

        # Cosine similarity between first two should be higher than with third
        sim_12 = np.dot(embeddings[0], embeddings[1])
        sim_13 = np.dot(embeddings[0], embeddings[2])

        assert sim_12 > sim_13

    def test_empty_list_raises_error(self):
        """Test that empty list raises an error."""
        with pytest.raises(EmbeddingError, match="No texts provided"):
            embed_texts([])
