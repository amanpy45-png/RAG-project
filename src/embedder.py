from langchain_ollama import OllamaEmbeddings


def load_embedding_model():
    embedding_model = OllamaEmbeddings(
        model="mxbai-embed-large"
    )

    return embedding_model