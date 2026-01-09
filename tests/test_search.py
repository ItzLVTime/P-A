from pdf_loader import load_pdf
from chunker import chunk_pages
from embeddings import embed_texts
from vector_store import build_index, search

pages = load_pdf("sample.pdf")
chunks = chunk_pages(pages)

texts = [c["text"] for c in chunks]
embeddings = embed_texts(texts)

index = build_index(embeddings)

query = "What is this document about?"
query_embedding = embed_texts([query])

ids = search(index, query_embedding, top_k=3)

print("Top matching chunks:")
for i in ids:
    print("Page:", chunks[i]["page"])
    print(chunks[i]["text"][:200])
    print("----")
