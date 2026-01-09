from backend.rag import build_context
from backend.llm import ask_llm

def answer_question(question, chunks, index):
    context, pages = build_context(chunks, index, question)

    prompt = f"""
You are a document assistant.
Answer ONLY using the context below.
If the answer is not present, say "Not found in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    answer = ask_llm(prompt)
    return answer.strip(), pages
