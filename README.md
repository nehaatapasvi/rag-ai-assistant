# RAG AI Assistant

A full Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, and Groq.  
Upload PDF documents and ask questions based on their content.

---

## Features

- Upload and process PDF documents
- Intelligent text chunking with overlap
- Semantic search using FAISS vector database
- Fast LLM responses using Groq (Llama 3.1)
- Local embeddings using sentence-transformers
- Context-aware answers grounded in document content

---

## Tech Stack

| Component       | Technology                          |
|----------------|------------------------------------|
| Frontend       | Streamlit                          |
| LLM            | Groq (Llama 3.1-8b-instant)        |
| Embeddings     | HuggingFace (all-MiniLM-L6-v2)     |
| Vector Store   | FAISS                              |
| PDF Processing | PyMuPDF                            |
| Framework      | LangChain                          |

---

## Prerequisites

- Python 3.10+
- Groq API key → https://console.groq.com/keys

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag-ai-assistant
```

### 2. Create virtual environment

```bash
python -m venv rag-env
source rag-env/bin/activate   
# Windows: rag-env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## Running the App

```bash
streamlit run app.py
```

App will be available at:

```
http://localhost:8501
```

---

## How It Works

1. PDF is uploaded and parsed using PyMuPDF  
2. Text is split into chunks  
3. Chunks are converted into embeddings  
4. Stored in FAISS vector database  
5. User query retrieves relevant chunks  
6. LLM generates answer using context  

---

## Configuration

You can modify these in `app.py`:

| Variable           | Description |
|------------------|------------|
| `LLM_MODEL`       | Groq model |
| `EMBEDDING_MODEL` | Embedding model |
| `CHUNK_SIZE`      | Chunk size |
| `CHUNK_OVERLAP`   | Overlap |

---

## Requirements

```
streamlit
langchain-community
langchain-core
langchain-huggingface
langchain-groq
pymupdf
faiss-cpu
sentence-transformers
python-dotenv
```

---

## License

MIT

---

## Acknowledgement

This project was built as part of a learning process exploring RAG systems and LLM applications.
