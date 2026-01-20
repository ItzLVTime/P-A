import pytest
import os
from backend.pdf_loader import load_pdf, PDFLoadError


class TestLoadPDF:
    def test_load_valid_pdf(self, sample_pdf_path):
        """Test loading a valid PDF file."""
        if not os.path.exists(sample_pdf_path):
            pytest.skip("sample.pdf not found")

        pages = load_pdf(sample_pdf_path)

        assert isinstance(pages, list)
        assert len(pages) > 0
        assert all("page" in p and "text" in p for p in pages)
        assert all(isinstance(p["page"], int) for p in pages)
        assert all(isinstance(p["text"], str) for p in pages)

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(PDFLoadError, match="File not found"):
            load_pdf("/nonexistent/path/file.pdf")

    def test_page_numbers_are_sequential(self, sample_pdf_path):
        """Test that page numbers start at 1 and are sequential."""
        if not os.path.exists(sample_pdf_path):
            pytest.skip("sample.pdf not found")

        pages = load_pdf(sample_pdf_path)
        page_numbers = [p["page"] for p in pages]

        assert page_numbers[0] >= 1
        for i in range(1, len(page_numbers)):
            assert page_numbers[i] > page_numbers[i - 1]
