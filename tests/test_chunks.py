from pdf_loader import load_pdf
from chunker import chunk_pages

pages = load_pdf("sample.pdf")
chunks = chunk_pages(pages)

print("Total chunks:", len(chunks))
print("First chunk page:", chunks[0]["page"])
print("First chunk text (preview):")
print(chunks[0]["text"][:300])
