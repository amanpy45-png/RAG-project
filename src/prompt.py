def create_prompt(query, results):
    """
    Create the prompt for the LLM.
    """

    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not found in the context,
say:

"I don't know based on the provided context."

Context:
{context}

Question:
{query}

Answer:
"""

    return prompt