"""Core RAG pipeline: load -> chunk -> embed/store -> retrieve -> answer."""
import os
import uuid
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

CHROMA_DIR = Path(".chroma")
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context. "
    "Cite sources as [filename p.N]. If the context does not contain the answer, "
    "say you don't know instead of guessing."
)


def load_pages(name: str, data: bytes | str) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for a PDF (bytes) or text file (str/bytes)."""
    if name.lower().endswith(".pdf"):
        import io
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    text = data.decode("utf-8", errors="ignore") if isinstance(data, bytes) else data
    return [(1, text)]


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Fixed-size character chunks with overlap."""
    if size <= overlap:
        raise ValueError("size must be greater than overlap")
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


class RAG:
    def __init__(self, persist_dir: Path = CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        self.col = self.client.get_or_create_collection(
            "docs", embedding_function=self.embed_fn
        )

    def ingest(self, name: str, data: bytes | str) -> int:
        ids, docs, metas = [], [], []
        for page, text in load_pages(name, data):
            for chunk in chunk_text(text):
                ids.append(str(uuid.uuid4()))
                docs.append(chunk)
                metas.append({"source": name, "page": page})
        if docs:
            self.col.add(ids=ids, documents=docs, metadatas=metas)
        return len(docs)

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        if self.col.count() == 0:
            return []
        res = self.col.query(query_texts=[query], n_results=k)
        return [
            {"text": d, **m, "distance": dist}
            for d, m, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def answer(self, query: str, k: int = 4) -> tuple[str, list[dict]]:
        hits = self.retrieve(query, k)
        if not hits:
            return "No documents indexed yet. Upload a file first.", []
        context = "\n\n".join(
            f"[{h['source']} p.{h['page']}]\n{h['text']}" for h in hits
        )
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=LLM_MODEL,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            }],
        )
        return msg.content[0].text, hits

    def reset(self):
        self.client.delete_collection("docs")
        self.col = self.client.get_or_create_collection(
            "docs", embedding_function=self.embed_fn
        )
