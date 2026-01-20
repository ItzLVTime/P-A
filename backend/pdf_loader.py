import fitz  # PyMuPDF
import os
import logging

logger = logging.getLogger(__name__)

class PDFLoadError(Exception):
    """Raised when PDF loading fails."""
    pass

def load_pdf(path, max_pages=500, max_size_mb=50):
    """
    Load and extract text from a PDF file.

    Args:
        path: Path to the PDF file
        max_pages: Maximum allowed pages (default 500)
        max_size_mb: Maximum file size in MB (default 50)

    Returns:
        List of dicts with 'page' and 'text' keys

    Raises:
        PDFLoadError: If file doesn't exist, is too large, or can't be parsed
    """
    logger.info(f"Loading PDF from: {path}")

    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise PDFLoadError(f"File not found: {path}")

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.debug(f"File size: {file_size_mb:.2f}MB")

    if file_size_mb > max_size_mb:
        logger.error(f"File too large: {file_size_mb:.1f}MB (max {max_size_mb}MB)")
        raise PDFLoadError(f"File too large: {file_size_mb:.1f}MB (max {max_size_mb}MB)")

    try:
        doc = fitz.open(path)
        logger.debug(f"PDF opened successfully, {doc.page_count} pages")
    except Exception as e:
        logger.error(f"Failed to open PDF: {e}")
        raise PDFLoadError(f"Failed to open PDF: {e}")

    if doc.page_count > max_pages:
        doc.close()
        logger.error(f"Too many pages: {doc.page_count} (max {max_pages})")
        raise PDFLoadError(f"Too many pages: {doc.page_count} (max {max_pages})")

    pages = []
    try:
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append({
                    "page": i + 1,
                    "text": text
                })
    except Exception as e:
        logger.error(f"Failed to extract text from page {i + 1}: {e}")
        raise PDFLoadError(f"Failed to extract text from page {i + 1}: {e}")
    finally:
        doc.close()

    if not pages:
        logger.error("No text content found in PDF")
        raise PDFLoadError("No text content found in PDF")

    logger.info(f"Successfully extracted {len(pages)} pages with text")
    return pages
