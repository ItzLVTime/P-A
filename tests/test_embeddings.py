from pdf_loader import load_pdf
from chunker import chunk_pages
from embeddings import embed_texts

pages = load_pdf("sample.pdf")
chunks = chunk_pages(pages)

texts = [c["text"] for c in chunks]
embeddings = embed_texts(texts)

print("Embeddings shape:", embeddings.shape)
