import streamlit as st
import os
from pathlib import Path
import pandas as pd
import tempfile
import json
import math

# Optional imports (only used when dependencies installed)
try:
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.docstore.document import Document
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain.llms import OpenAI
    import faiss
    import PyPDF2
    DEP_AVAILABLE = True
except Exception as e:
    DEP_AVAILABLE = False

st.set_page_config(page_title="PiRAG — RAG + FAISS Demo", layout="wide")

st.title("PiRAG — Minimal Real RAG + FAISS Flow (Demo)")
st.markdown("""
This demo contains a lightweight Retrieval-Augmented Generation (RAG) flow demonstrating:
- Upload PDF / TXT SOPs → extract text → chunk → embed → store in FAISS
- Ask PiRAG → retrieve top-k docs → LLM answer with inline citations

**Notes:** This Streamlit app is shipped as code only. To run the retrieval/embedding code you need the dependencies in `requirements.txt` and an OpenAI API key (or swap embeddings/LLM to another provider). If dependencies are missing the app will show instructions.
""")

KB_DIR = Path(__file__).parent / "knowledge_base"
KB_DIR.mkdir(exist_ok=True)

st.sidebar.header("RAG Controls")
uploaded = st.sidebar.file_uploader("Upload SOP / PDF / TXT to add to KB", type=["pdf","txt","md"])
if st.sidebar.button("Ingest uploaded file"):
    if uploaded is None:
        st.sidebar.error("Please choose a file to upload first.")
    else:
        out_path = KB_DIR / uploaded.name
        with open(out_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.sidebar.success(f"Saved to knowledge_base/{uploaded.name} — now click 'Build or Rebuild Index' to (re)index.")

st.sidebar.markdown("---")
st.sidebar.write("Indexing / Retrieval")
build_index = st.sidebar.button("Build or Rebuild Index (FAISS)")
top_k = st.sidebar.slider("Top k retrieved", min_value=1, max_value=10, value=3)

if not DEP_AVAILABLE:
    st.warning("Optional dependencies for RAG are not installed in this environment. The code is present in this demo package; install the packages in requirements.txt and set OPENAI_API_KEY to run embeddings/LLM.")
    st.info("Required (example): pip install -r requirements.txt\nSet OPENAI_API_KEY environment variable before running streamlit. The example uses OpenAI embeddings + OpenAI LLM.")
    st.stop()

# When dependencies available -- real flow
st.success("Dependencies available — RAG code will run.")

# Utility: load docs
def load_documents_from_folder(folder: Path):
    docs = []
    for p in sorted(folder.glob("*")):
        if p.suffix.lower() in [".txt", ".md"]:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append({"source": str(p.name), "text": text})
        elif p.suffix.lower() == ".pdf":
            try:
                with open(p, "rb") as fh:
                    reader = PyPDF2.PdfReader(fh)
                    text = []
                    for page in reader.pages:
                        try:
                            t = page.extract_text() or ""
                        except:
                            t = ""
                        text.append(t)
                    text = "\n".join(text)
                docs.append({"source": str(p.name), "text": text})
            except Exception as e:
                st.error(f"Failed to read PDF {p.name}: {e}")
    return docs

# Simple chunker (split by paragraphs and then into ~1000 char chunks)
def chunk_text(text, chunk_size=1000, overlap=200):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                start = end - overlap
    return chunks

# Build / Rebuild FAISS index
INDEX_DIR = Path(__file__).parent / "vector_index"
INDEX_DIR.mkdir(exist_ok=True)
INDEX_PATH = INDEX_DIR / "faiss_index"

if build_index:
    st.info("Loading documents from knowledge_base/...")
    docs = load_documents_from_folder(KB_DIR)
    if len(docs) == 0:
        st.warning("No documents found in knowledge_base — add files via sidebar upload or place sample SOPs in the knowledge_base folder.")
    else:
        # Prepare texts and metadata
        texts = []
        metadatas = []
        for d in docs:
            chunks = chunk_text(d["text"], chunk_size=800, overlap=150)
            for i,ch in enumerate(chunks):
                texts.append(ch)
                metadatas.append({"source": d["source"], "chunk": i})
        st.write(f"Prepared {len(texts)} chunks from {len(docs)} documents.")

        # Create embeddings and FAISS index
        st.info("Creating embeddings (OpenAI Embeddings expected via OPENAI_API_KEY) and FAISS index...")
        embed = OpenAIEmbeddings()
        # Build vectorstore
        try:
            vs = FAISS.from_texts(texts, embed, metadatas=metadatas)
            # persist
            vs.save_local(str(INDEX_PATH))
            st.success("Index built and saved to vector_index/")
        except Exception as e:
            st.error(f"Failed to build FAISS index: {e}")


# Query interface
st.markdown("## Ask PiRAG (RAG-powered)")
query = st.text_input("Ask PiRAG a question about the paint shop / SOPs", value="What are the standard rework steps for paint runs with sagging?")
if st.button("Run Query"):
    if not INDEX_PATH.exists():
        st.error("Index not found. Build the index first (sidebar).")
    else:
        st.info("Loading index...")
        embed = OpenAIEmbeddings()
        vs = FAISS.load_local(str(INDEX_PATH), embed)
        docs_and_scores = vs.similarity_search_with_score(query, k=top_k)
        st.write("Top retrieved chunks (source — chunk # and similarity score):")
        for d,score in docs_and_scores:
            st.markdown(f"- **{d.metadata.get('source','?')}** — chunk {d.metadata.get('chunk','?')} — score: {score:.4f}")
            st.write(d.page_content[:800]+"...")

        # Run an LLM chain with sources (LangChain QA-with-sources)
        try:
            llm = OpenAI(temperature=0)
            chain = load_qa_with_sources_chain(llm, chain_type="stuff")
            docs_for_chain = [Document(page_content=d.page_content, metadata=d.metadata) for d,sc in docs_and_scores]
            answer = chain.run(input_documents=docs_for_chain, question=query)
            st.markdown("### PiRAG Answer (with cited sources)")
            st.write(answer)
        except Exception as e:
            st.error(f"LLM chain failed: {e}")

st.markdown("---")
st.markdown("### Knowledge base contents (files in `knowledge_base` folder):")
for p in sorted(KB_DIR.glob("*")):
    st.write(f"- {p.name} ({p.stat().st_size} bytes)")
