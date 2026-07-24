import os
from datetime import datetime
import streamlit as st
from langchain_ollama import ChatOllama
from src.loader import load_pdf
from src.text_splitter import split_documents
from src.embedder import load_embedding_model
from src.faiss_vector_store import create_vector_store, retrieve_documents
from src.prompt import create_prompt

MODEL_NAME = "llama3.1"
st.set_page_config(page_title="PRISM · Local RAG", page_icon="◆", layout="centered")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0e12;--surf:#12171d;--surf2:#1a212a;--text:#e8ecef;--dim:#6b7684;--line:#232b34;--signal:#4fd1a5;}
.stApp{background:var(--bg);} .block-container{padding-top:1.2rem;max-width:760px;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:var(--text);}
.hdr{display:flex;align-items:center;gap:14px;}
.mark{width:38px;height:38px;border-radius:9px;background:linear-gradient(155deg,var(--signal),#2a8a6c);
  display:flex;align-items:center;justify-content:center;font-family:'Space Grotesk',sans-serif;font-weight:700;color:#06110d;}
.hdr h1{font-family:'Space Grotesk',sans-serif;font-size:26px;margin:0;letter-spacing:-.01em;}
.hdr p{margin:2px 0 0;font-size:13.5px;color:var(--dim);}
.strip{display:flex;margin:18px 0 22px;border:1px solid var(--line);border-radius:10px;overflow:hidden;
  background:var(--surf);font-family:'JetBrains Mono',monospace;font-size:12px;}
.cell{flex:1;padding:10px 14px;border-right:1px solid var(--line);}
.cell:last-child{border-right:none;}
.lbl{color:var(--dim);font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px;}
.val{display:flex;align-items:center;gap:6px;}
.dot{width:7px;height:7px;border-radius:50%;background:var(--signal);box-shadow:0 0 6px var(--signal);}
.dot.idle{background:var(--dim);box-shadow:none;}
section[data-testid="stSidebar"]{background:var(--surf);border-right:1px solid var(--line);}
.eyebrow{font-family:'JetBrains Mono',monospace;font-size:11px;text-transform:uppercase;color:var(--dim);letter-spacing:.08em;}
.doc-log{font-family:'JetBrains Mono',monospace;font-size:12px;background:var(--surf2);border:1px solid var(--line);
  border-left:2px solid var(--signal);border-radius:6px;padding:10px 12px;margin-top:10px;line-height:1.6;}
.doc-log .k{color:var(--dim);}
[data-testid="stFileUploader"] section{background:var(--surf2);border:1px dashed var(--line);border-radius:10px;}
[data-testid="stChatMessage"]{background:var(--surf);border:1px solid var(--line);border-radius:12px;}
.empty{text-align:center;padding:56px 20px;border:1px dashed var(--line);border-radius:10px;color:var(--dim);}
.empty .g{font-family:'Space Grotesk',sans-serif;font-size:30px;color:var(--line);margin-bottom:10px;}
.empty h3{font-family:'Space Grotesk',sans-serif;color:var(--text);font-size:16px;margin:0 0 4px;}
.stButton>button{background:var(--surf2);border:1px solid var(--line);color:var(--text);border-radius:8px;}
.stButton>button:hover{border-color:#f0765a;color:#f0765a;}
</style>
""", unsafe_allow_html=True)

for k, v in {"vector_store": None, "messages": [], "doc_meta": None}.items():
    st.session_state.setdefault(k, v)

st.markdown("""
<div class="hdr"><div class="mark">P</div>
<div><h1>PRISM</h1><p>PDF Retrieval & Inference on a Secure Machine — fully offline</p></div></div>
""", unsafe_allow_html=True)

ready = st.session_state.vector_store is not None
idx_txt = f"{st.session_state.doc_meta['chunks']} chunks indexed" if ready else "waiting for upload"
st.markdown(f"""
<div class="strip">
  <div class="cell"><div class="lbl">Runtime</div><div class="val"><span class="dot"></span>Local · Offline</div></div>
  <div class="cell"><div class="lbl">Model</div><div class="val"><span class="dot"></span>{MODEL_NAME}</div></div>
  <div class="cell"><div class="lbl">Index</div><div class="val"><span class="dot {'' if ready else 'idle'}"></span>{idx_txt}</div></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="eyebrow">Document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file and (not st.session_state.doc_meta or st.session_state.doc_meta["name"] != uploaded_file.name):
        os.makedirs("data", exist_ok=True)
        path = os.path.join("data", uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Reading, chunking, embedding..."):
            chunks = split_documents(load_pdf(path))
            st.session_state.vector_store = create_vector_store(chunks, load_embedding_model())
        st.session_state.doc_meta = {
            "name": uploaded_file.name,
            "size_kb": round(len(uploaded_file.getbuffer()) / 1024, 1),
            "chunks": len(chunks),
            "time": datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state.messages = []
        st.rerun()

    if st.session_state.doc_meta:
        m = st.session_state.doc_meta
        st.markdown(f"""<div class="doc-log">
        <span class="k">file</span> {m['name']}<br><span class="k">size</span> {m['size_kb']} KB<br>
        <span class="k">chunks</span> {m['chunks']}<br><span class="k">ready</span> {m['time']}</div>""",
        unsafe_allow_html=True)

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("About"):
        st.write("Runs locally via **Ollama** (`llama3.1`) + **FAISS**. Nothing leaves your device.")

if not st.session_state.messages:
    msg = "Document is ready — ask anything about it." if ready else "Upload a PDF from the sidebar to start."
    st.markdown(f'<div class="empty"><div class="g">◆</div><h3>{"Ready" if ready else "No document loaded"}</h3><p>{msg}</p></div>', unsafe_allow_html=True)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask a question about your PDF..."):
    if not ready:
        st.warning("Upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("retrieving context · generating locally..."):
                results = retrieve_documents(st.session_state.vector_store, prompt)
                rag_prompt = create_prompt(prompt, results)
                answer = ChatOllama(model=MODEL_NAME, temperature=0).invoke(rag_prompt).content
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})