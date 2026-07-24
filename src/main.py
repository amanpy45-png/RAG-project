from langchain_ollama import ChatOllama

from loader import load_pdf
from text_splitter import split_documents
from embedder import load_embedding_model
from faiss_vector_store import (
    create_vector_store,
    retrieve_documents
)
from prompt import create_prompt


def main():
    # Step 1
    documents = load_pdf("data/sample.pdf")

    # Step 2
    chunks = split_documents(documents)

    # Step 3
    embedding_model = load_embedding_model()

    # Step 4
    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    # Step 5 
    llm = ChatOllama(
        model="llama3.1",
        temperature=0
    )
    while True:

        query = input("\nAsk your question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        # Step 6
        results = retrieve_documents(
            vector_store,
            query
        )
        # Step 7
        prompt = create_prompt(
            query,
            results
        )
        # Step 8
        response = llm.invoke(prompt)

        print("\nAnswer:\n")
        print(response.content)

if __name__ == "__main__":
    main()