from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):

    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )
    return vector_store

def retrieve_documents(vector_store,
                       query,
                       k=3):

    results = vector_store.similarity_search(
        query,
        k=k
    )
    return results