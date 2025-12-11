A simple RAG system using TF-IDF + ChromaDB for retrieval and Groq LLM for answering. Runs on any CPU machine with no heavy ML libraries.

## How to Use
1. Ingest PDFs and build vector DB:
   python rag_ingestion.py

2. Ask questions:
   python rag_retrieval.py

## Install
pip install -r requirements.txt

## Set Groq Key
export GROQ_API_KEY="your_api_key"

## Notes
- Uses TF-IDF instead of heavy embeddings.
- Fast, lightweight, and easy to run.
