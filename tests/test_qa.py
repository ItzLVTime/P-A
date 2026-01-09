from pdf_loader import load_pdf
from chunker import chunk_pages
from embeddings import embed_texts
from vector_store import build_index
from qa_engine import answer_question

# Load and prepare document
pages = load_pdf("sample.pdf")
chunks = chunk_pages(pages)

texts = [c["text"] for c in chunks]
embeddings = embed_texts(texts)
index = build_index(embeddings)

# Ask a question
question = "What is this document about?"
answer, pages_used = answer_question(question, chunks, index)

print("ANSWER:")
print(answer)
print("\nPAGES:", pages_used)
