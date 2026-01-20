import pytest
import numpy as np
from backend.embeddings import embed_texts
from backend.vector_store import build_index, search, VectorStoreError


class TestVectorStore:
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        texts = [
            "Python programming language",
            "Machine learning with Python",
            "JavaScript web development",
            "Data science and analytics",
        ]
        return embed_texts(texts)

    @pytest.fixture
    def sample_index(self, sample_embeddings):
        """Create a sample FAISS index."""
        return build_index(sample_embeddings)

    def test_build_index(self, sample_embeddings):
        """Test building a FAISS index."""
        index = build_index(sample_embeddings)
        assert index is not None
        assert index.ntotal == len(sample_embeddings)

    def test_search_returns_correct_count(self, sample_index):
        """Test that search returns the requested number of results."""
        query_embedding = embed_texts(["Python code"])

        results = search(sample_index, query_embedding, top_k=2)
        assert len(results) == 2

        results = search(sample_index, query_embedding, top_k=3)
        assert len(results) == 3

    def test_search_returns_valid_indices(self, sample_index, sample_embeddings):
        """Test that search returns valid indices."""
        query_embedding = embed_texts(["Python"])
        results = search(sample_index, query_embedding, top_k=2)

        for idx in results:
            assert 0 <= idx < len(sample_embeddings)

    def test_search_finds_relevant_results(self, sample_index):
        """Test that search finds semantically relevant results."""
        # Query about Python should return Python-related indices (0 or 1)
        query_embedding = embed_texts(["Python programming"])
        results = search(sample_index, query_embedding, top_k=2)

        # At least one of top 2 should be index 0 or 1 (Python-related)
        assert 0 in results or 1 in results

    def test_build_index_empty_raises_error(self):
        """Test that building index with empty embeddings raises error."""
        with pytest.raises(VectorStoreError):
            build_index(np.array([]))

    def test_search_none_index_raises_error(self):
        """Test that searching with None index raises error."""
        query_embedding = embed_texts(["test"])
        with pytest.raises(VectorStoreError, match="Index not initialized"):
            search(None, query_embedding)
