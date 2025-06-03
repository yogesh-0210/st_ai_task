# Document Q&A AI Agent

This project is an AI-powered agent for question answering over PDF documents and searching academic papers on arXiv. It uses language models, vector databases, and embeddings to provide answers to user queries about uploaded documents.

## Features

- Extracts text from PDF files using PyMuPDF.
- Splits and embeds document text using Sentence Transformers and stores them in ChromaDB.
- Answers questions about the uploaded documents using a language model (Flan-T5 via HuggingFace).
- Searches for academic papers on arXiv and displays their metadata.
- Provides both a Streamlit web interface and a CLI fallback.

## Requirements

See [Requirements.txt](Requirements.txt) for the full list. Key dependencies include:
- langchain
- chromadb
- sentence-transformers
- streamlit
- PyMuPDF
- arxiv
- huggingface-hub

Install dependencies with:
```sh
pip install -r Requirements.txt
```

## Usage

### Streamlit Web App

To launch the web interface:
```sh
streamlit run document_qa_agent.py
```
- Upload one or more PDF files.
- Ask questions about the documents.
- Search for papers on arXiv.

### Command Line Interface

If Streamlit is not installed, the script will run in CLI mode:
```sh
python document_qa_agent.py
```
- Edit the `file_paths` list in the script to point to your PDF files.
- Enter your questions and arXiv search queries when prompted.

## File Structure

- `document_qa_agent.py` - Main application script.
- `Requirements.txt` - Python dependencies.

## Demo

A sample demo video is included as `Demo.mp4`.
