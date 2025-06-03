# document_qa_agent.py

import fitz  # PyMuPDF
import arxiv
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import os
import torch
os.environ["STREAMLIT_WATCH_DISABLE"] = "true"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cpu')
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- PDF Text Extraction ---
def extract_text_from_pdf(pdf_path):
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()  # this returns a string
    return full_text


# --- Chunking and Embedding with ChromaDB ---
def store_in_vector_db(texts, persist_dir="chroma_db"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(texts)]
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding_model, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

# --- Vector Similarity QA ---
def qa_with_vector_db(vectordb, question):
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=HuggingFaceHub(repo_id="google/flan-t5-large"),
        chain_type="stuff",
        retriever=retriever
    )
    return qa_chain.run(question)

# --- Arxiv API Search ---
def search_arxiv(query):
    results = arxiv.Search(query=query, max_results=1).results()
    for result in results:
        return {
            "title": result.title,
            "url": result.entry_id,
            "pdf": result.pdf_url,
            "summary": result.summary
        }

# --- Streamlit Interface ---
try:
    import streamlit as st
    USE_STREAMLIT = True
except ImportError:
    USE_STREAMLIT = False

if USE_STREAMLIT:
    st.title("Document Q&A AI Agent")

    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)
    if uploaded_files:
        all_text = ""
        for file in uploaded_files:
            with open(file.name, "wb") as f:
                f.write(file.read())
            all_text += extract_text_from_pdf(file.name)

        st.write("Indexing documents...")
        vectordb = store_in_vector_db(all_text)

        question = st.text_input("Ask a question about the documents")
        if question:
            answer = qa_with_vector_db(vectordb, question)
            st.write("Answer:", answer)

        search_query = st.text_input("Search a paper on arXiv")
        if search_query:
            result = search_arxiv(search_query)
            st.write(result)
else:
    # CLI fallback
    if __name__ == "__main__":
        file_paths = ["sample1.pdf", "sample2.pdf"]
        all_text = ""
        for path in file_paths:
            all_text += extract_text_from_pdf(path)

        vectordb = store_in_vector_db(all_text)

        user_query = input("Ask a question: ")
        print("Answer:", qa_with_vector_db(vectordb, user_query))

        arxiv_q = input("Search a paper on arXiv: ")
        print(search_arxiv(arxiv_q))
