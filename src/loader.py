# PDF
#    │
#    ▼
# PyPDFLoader
#    │
#    ▼
# List[Document]
#    │
#    ├── page_content  ← Actual text
#    └── metadata      ← Source, page number, etc.


from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents
