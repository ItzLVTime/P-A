# P&A – PDF Question Answering System

P&A is a fully local AI-based PDF Question Answering application that allows users to upload a PDF and ask questions strictly based on its content.

## Features
- PDF upload and processing
- Semantic search using embeddings
- Accurate answers using a local LLM
- Source page references
- Fully offline / local execution

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
```

### 2. Install the requirements
```bash
pip install -r requirements.txt
```

### 3. Start LLM Server
```bash
litellm --model ollama/qwen2.5:7b-instruct-q4_k_m --port 8000
```

### 4. Run the app
```bash
streamlit run app.py
```

## Notes
- Ensure Ollama is installed and running before starting LiteLLM.
- Default model: Qwen 2.5 7B Instruct (Q4_K_M)
- Recommended: 6GB+ VRAM
