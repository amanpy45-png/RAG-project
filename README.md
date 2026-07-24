# RAG Chatbot from Scratch

A Retrieval-Augmented Generation (RAG) chatbot built from scratch using **LangChain**, **FAISS**, **Sentence Transformers**, and a **Local LLM (Ollama)**.

The goal of this project was **to understand every component of a RAG pipeline instead of relying on high-level abstractions**.

---

## Features

-  Load PDF documents
-  Split documents into overlapping chunks
-  Generate embeddings using Sentence Transformers
-  Store embeddings in FAISS Vector Database
-  Semantic similarity search
-  Generate answers using a Local LLM (Ollama)
-  Interactive command-line chatbot

---

##  Architecture

```text
                    User Question
                          │
                          ▼
                  Similarity Search
                          │
                          ▼
               Top-k Relevant Chunks
                          │
                          ▼
                 Prompt Construction
                          │
                          ▼
                     Local LLM
                     (Ollama)
                          │
                          ▼
                    Final Answer
```

---

##  Project Structure

```text
rag-chatbot/
│
├── data/
│   └── sample.pdf
│
├── loader.py
├── text_splitter.py
├── embedder.py
├── faiss_vector_store.py
├── prompt_builder.py
├── main.py
│
├── requirements.txt
└── README.md
```

---

##  Tech Stack

- Python
- LangChain
- FAISS
- Sentence Transformers
- Ollama
- Llama 3.1 (Local LLM)
- PyPDF

---

##  RAG Pipeline

### 1. Load PDF

```python
documents = load_pdf("data/sample.pdf")
```

---

### 2. Split Documents

```python
chunks = split_documents(documents)
```

---

### 3. Generate Embeddings

```python
embedding_model = load_embedding_model()
```

---

### 4. Create FAISS Vector Store

```python
vector_store = create_vector_store(
    chunks,
    embedding_model
)
```

---

### 5. Retrieve Relevant Documents

```python
results = retrieve_documents(
    vector_store,
    query,
    k=3
)
```

---

### 6. Build Prompt

```python
prompt = build_prompt(
    results,
    query
)
```

---

### 7. Generate Response

```python
answer = generate_answer(
    llm,
    prompt
)
```

---

##  Installation

### Clone Repository

```bash
git clone https://github.com/amanpy45-png/rag-chatbot.git

cd rag-chatbot
```

---

### Create Virtual Environment

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Install Ollama

Download Ollama:

https://ollama.com/download

Pull a model:

```bash
ollama pull llama3.1
```

or

```bash
ollama pull qwen3:4b
```

Run Ollama:

```bash
ollama serve
```

---

##  Run Project

```bash
python main.py
```

Example

```text
Ask your question:
What is Retrieval-Augmented Generation?

Answer:
Retrieval-Augmented Generation (RAG) combines information retrieval with
large language models to answer questions using external knowledge.
```

---

##  Concepts Learned

- Retrieval-Augmented Generation (RAG)
- Document Loading
- Document Chunking
- Embeddings
- Vector Databases
- FAISS
- Semantic Search
- Prompt Engineering
- Local LLM Inference
- LangChain Basics

---

##  Learning Objectives

This project was built to understand the internal workflow of RAG systems by implementing each step manually before using higher-level LangChain abstractions.

Topics covered include:

- How embeddings work
- Why chunking is required
- Semantic similarity search
- FAISS indexing
- Prompt construction
- LLM response generation

---

##  Author

**Aman Negi**

GitHub: https://github.com/amanpy45-png

LinkedIn: https://linkedin.com/in/amanpy54

---

##  Acknowledgements

This project was created as part of my journey to learn **Generative AI**, **Retrieval-Augmented Generation (RAG)**, and **LangChain** through hands-on implementation rather than relying on pre-built abstractions.

---

⭐ If you found this project useful, consider giving it a star!
