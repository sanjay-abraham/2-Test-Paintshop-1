PiRAG — RAG + FAISS Demo Package
================================

Contents:
- PiRAG_Streamlit_App_RAG.py  : Streamlit app demonstrating a minimal RAG + FAISS flow.
- knowledge_base/             : Example SOPs, RCA notes, ESG norms (text files).
- requirements.txt            : Suggested environment packages.
- docs/                       : Additional PDFs and SOP markdowns for partners.
- run_demo.sh                 : Simple run script (uses streamlit).

How to run (example):
1. Create a python venv: python -m venv .venv
2. pip install -r requirements.txt
3. export OPENAI_API_KEY=<your_key>
4. streamlit run PiRAG_Streamlit_App_RAG.py

Notes:
- The app uses OpenAI embeddings + OpenAI LLM by default via LangChain. You can swap embeddings to any provider supported by LangChain.
- If you don't want to use OpenAI, update the embeddings/LLM section in the app file.