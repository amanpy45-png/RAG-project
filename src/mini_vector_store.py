import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


from src.chunker import chunks
from src.embedder import embeddings

model = SentenceTransformer("all-MiniLM-L6-v2")

vector_store = list(zip(embeddings, chunks))

document_embeddings = np.array([item[0] for item in vector_store])
chunks = [item[1] for item in vector_store]

def retrieve(query, k=3):
    query_embedding = model.encode(query)

    query_embedding = query_embedding.reshape(1, -1)

    similarities = cosine_similarity(query_embedding, document_embeddings)

    scores = similarities[0]

    top_k = np.argsort(scores)[::-1][:k]

    return [vector_store[i][1] for i in top_k]