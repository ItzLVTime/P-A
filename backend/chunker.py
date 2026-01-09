def chunk_pages(pages, chunk_size=500, overlap=100):
    chunks = []

    for page in pages:
        words = page["text"].split()
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])

            chunks.append({
                "page": page["page"],
                "text": chunk_text
            })

            start += chunk_size - overlap

    return chunks
