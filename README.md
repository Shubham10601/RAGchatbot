# RAGchatbot
A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDFs and ask questions using LangChain, FAISS, HuggingFace embeddings, and Groq LLM.
# PDF RAG Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content.

## Features

- Upload PDF documents
- Extract and process text
- Semantic search using vector embeddings
- AI-generated answers from document context
- Built using open-source and free AI models

## Tech Stack

- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq LLM
- PDFPlumber

## Model Used

- Embedding Model:
  - sentence-transformers/all-MiniLM-L6-v2

- LLM:
  - llama-3.1-8b-instant

## Installation

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run chatbot.py
```

## Future Improvements

- Conversational memory
- Multi-PDF support
- Chat history
- Better retrieval optimization
- Deployment support
- Authentication system

## Author

Shubham Maurya
