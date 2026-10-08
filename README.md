# 📚 RAG Chatbot: Chat with Your Documents

Upload PDFs/text files and ask questions. Answers are grounded in your documents
and cite their sources (`[file p.N]`); if the answer isn't in the context, the bot says so.

## Architecture
```
PDF/TXT -> chunk (800 chars, 150 overlap) -> embed (MiniLM, local) -> ChromaDB
Question -> embed -> top-k retrieval -> prompt + context -> Claude -> cited answer
```

## Stack
Python · Streamlit · ChromaDB · sentence-transformers · Anthropic API

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # add your ANTHROPIC_API_KEY
streamlit run app.py
pytest
```

## Design decisions
- Local embeddings: free, private, no API key for retrieval.
- Overlapping chunks preserve context across boundaries.
- Strict system prompt reduces hallucination.

## Roadmap
- [ ] Hybrid search (BM25 + vector) and reranking
- [ ] Retrieval evaluation set (hit rate / MRR)
- [ ] Streaming answers, Docker image, deploy to Hugging Face Spaces
