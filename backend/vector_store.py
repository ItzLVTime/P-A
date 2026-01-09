import faiss
import numpy as np

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search(index, query_embedding, top_k=3):
    distances, indices = index.search(query_embedding, top_k)
    return indices[0]
