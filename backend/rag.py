from backend.embeddings import embed_texts
from backend.vector_store import search

def build_context(chunks, index, question, top_k=3):
    question_embedding = embed_texts([question])
    ids = search(index, question_embedding, top_k)

    context_blocks = []
    pages = set()

    for i in ids:
        context_blocks.append(chunks[i]["text"])
        pages.add(chunks[i]["page"])

    context = "\n\n".join(context_blocks)
    return context, sorted(pages)
