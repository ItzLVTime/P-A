import pytest
from unittest.mock import patch, MagicMock
from backend.qa_engine import answer_question, QAError
from backend.embeddings import embed_texts
from backend.vector_store import build_index


class TestQAEngine:
    @pytest.fixture
    def qa_setup(self):
        """Set up chunks and index for Q&A testing."""
        chunks = [
            {"page": 1, "text": "Python is a programming language created by Guido van Rossum."},
            {"page": 2, "text": "Machine learning uses algorithms to learn from data."},
            {"page": 3, "text": "Data science combines statistics and programming."},
        ]
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        index = build_index(embeddings)
        return chunks, index

    def test_answer_question_returns_tuple(self, qa_setup):
        """Test that answer_question returns a tuple of (answer, pages)."""
        chunks, index = qa_setup

        with patch("backend.qa_engine.ask_llm") as mock_llm:
            mock_llm.return_value = "Python was created by Guido van Rossum."

            answer, pages = answer_question("Who created Python?", chunks, index)

            assert isinstance(answer, str)
            assert isinstance(pages, list)
            assert len(answer) > 0

    def test_answer_includes_source_pages(self, qa_setup):
        """Test that source pages are returned."""
        chunks, index = qa_setup

        with patch("backend.qa_engine.ask_llm") as mock_llm:
            mock_llm.return_value = "Answer from document"

            _, pages = answer_question("What is Python?", chunks, index)

            assert len(pages) > 0
            assert all(isinstance(p, int) for p in pages)

    def test_top_k_parameter(self, qa_setup):
        """Test that top_k parameter affects number of source pages."""
        chunks, index = qa_setup

        with patch("backend.qa_engine.ask_llm") as mock_llm:
            mock_llm.return_value = "Answer"

            _, pages_k1 = answer_question("test", chunks, index, top_k=1)
            _, pages_k3 = answer_question("test", chunks, index, top_k=3)

            assert len(pages_k1) <= len(pages_k3)

    def test_llm_error_raises_qa_error(self, qa_setup):
        """Test that LLM errors are wrapped in QAError."""
        chunks, index = qa_setup

        with patch("backend.qa_engine.ask_llm") as mock_llm:
            from backend.llm import LLMError
            mock_llm.side_effect = LLMError("API failed")

            with pytest.raises(QAError, match="Failed to get answer from LLM"):
                answer_question("test question", chunks, index)

    def test_empty_chunks_raises_error(self):
        """Test that empty chunks raises an error."""
        with pytest.raises(QAError):
            answer_question("test", [], None)
