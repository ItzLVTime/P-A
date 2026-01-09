# P&A – PDF Question Answering System

P&A is a local AI-based PDF Question Answering application that allows users to upload a PDF and ask questions based strictly on its content.

## Features
- PDF upload and processing
- Semantic search with embeddings
- Accurate answers using local LLM
- Source page references
- Fully local execution

## Tech Stack
- Python
- Streamlit
- FAISS
- Sentence Transformers
- Ollama + LiteLLM
- Qwen 2.5 7B Instruct (default)

## Setup

### 1. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate

## 2. Install the requirements
pip install -r requirements.txt

## 3. Start LLM Server
litellm --model ollama/qwen2.5:7b-instruct-q4_k_m --port 8000

## 4. Run the app
streamlit run app.py

