import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_pdf_path():
    """Path to sample PDF for testing."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample.pdf")

@pytest.fixture
def sample_pages():
    """Sample page data for testing with proper sentences."""
    return [
        {"page": 1, "text": "This is the first page with some content about Python programming. Python is a versatile language used for web development. It supports multiple programming paradigms. Many developers love Python for its simplicity."},
        {"page": 2, "text": "The second page discusses machine learning and AI concepts. Machine learning enables computers to learn from data. Deep learning is a subset of machine learning. Neural networks are inspired by the human brain."},
        {"page": 3, "text": "Final page covers data science and analytics topics. Data science combines statistics and programming. Analytics helps businesses make better decisions. Visualization is key to understanding data."},
    ]

@pytest.fixture
def sample_chunks(sample_pages):
    """Sample chunks for testing."""
    from backend.chunker import chunk_pages
    return chunk_pages(sample_pages)
