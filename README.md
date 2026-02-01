# RAG ChatBot

A document Q&A app: upload a PDF, ask questions, get answers grounded in the text. Uses RAG (retrieval-augmented generation) with FAISS and supports **Ollama** (local) or **OpenAI**.

## Features

- **Ollama (local):** tinyllama (chat), embeddinggemma (embeddings)
- **OpenAI:** gpt-4o-mini / gpt-4o / gpt-4-turbo and text embeddings
- Chat-style UI with quick-question buttons and conversation history
- New chat when you upload a different document

## Prerequisites

- **Ollama:** [Install Ollama](https://ollama.ai), then run:
  ```bash
  ollama pull tinyllama
  ollama pull embeddinggemma
  ```
- **OpenAI (optional):** API key from [OpenAI](https://platform.openai.com)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run ragChatBot.py
```

Open the URL in your browser (usually `http://localhost:8501`).

## Usage

1. In the sidebar, choose **Ollama (local)** or **OpenAI** and pick chat/embedding models.
2. For OpenAI, enter your API key (or set `OPENAI_API_KEY`).
3. Upload a **non-OCR PDF**.
4. Use the quick questions or type your own in the chat input.

Answers are based only on the uploaded document.
