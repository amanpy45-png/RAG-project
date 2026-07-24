# Paragraph
# ↓
# Newline
# ↓
# Space
# ↓
# Character

# splitter prefers to split at higher-priority separators.

# It is not an exact sliding-window chunker.
# It prioritizes natural text boundaries over perfectly consistent overlap.





from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents,
                    chunk_size=500,
                    chunk_overlap=100):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    return chunks