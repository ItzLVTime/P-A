import pytest
from backend.chunker import chunk_pages


class TestChunkPages:
    def test_chunk_pages_returns_list(self, sample_pages):
        """Test that chunk_pages returns a list."""
        chunks = chunk_pages(sample_pages)
        assert isinstance(chunks, list)

    def test_chunks_have_required_keys(self, sample_pages):
        """Test that each chunk has page and text keys."""
        chunks = chunk_pages(sample_pages)

        for chunk in chunks:
            assert "page" in chunk
            assert "text" in chunk
            assert isinstance(chunk["page"], int)
            assert isinstance(chunk["text"], str)

    def test_chunk_size_parameter(self, sample_pages):
        """Test that chunk_size parameter affects output."""
        small_chunks = chunk_pages(sample_pages, chunk_size=5)
        large_chunks = chunk_pages(sample_pages, chunk_size=100)

        # Smaller chunk size should produce more chunks
        assert len(small_chunks) >= len(large_chunks)

    def test_overlap_parameter(self, sample_pages):
        """Test that overlap parameter works."""
        no_overlap = chunk_pages(sample_pages, chunk_size=10, overlap=0)
        with_overlap = chunk_pages(sample_pages, chunk_size=10, overlap=5)

        # With overlap, we should have more chunks
        assert len(with_overlap) >= len(no_overlap)

    def test_empty_pages_list(self):
        """Test handling of empty pages list."""
        chunks = chunk_pages([])
        assert chunks == []

    def test_preserves_page_numbers(self, sample_pages):
        """Test that page numbers are preserved in chunks."""
        chunks = chunk_pages(sample_pages)
        chunk_pages_set = {c["page"] for c in chunks}
        original_pages_set = {p["page"] for p in sample_pages}

        assert chunk_pages_set == original_pages_set
