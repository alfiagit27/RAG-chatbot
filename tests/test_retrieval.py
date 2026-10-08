"""Retrieval smoke test: no API key needed (embeddings are local)."""
import pytest
from rag import RAG

DOC = (
    "The Eiffel Tower is located in Paris, France. It was completed in 1889. "
    + "Filler text about unrelated gardening and soil nutrients. " * 40
    + "Python was created by Guido van Rossum and first released in 1991. "
    + "More filler about cooking pasta and boiling water. " * 40
    + "The mitochondria is the organelle that produces most of a cell's ATP."
)


@pytest.fixture
def rag(tmp_path):
    r = RAG(persist_dir=tmp_path)
    r.ingest("sample.txt", DOC)
    return r


@pytest.mark.parametrize("query,expected", [
    ("Who created Python?", "Guido van Rossum"),
    ("When was the Eiffel Tower finished?", "1889"),
    ("Which organelle makes ATP?", "mitochondria"),
])
def test_relevant_chunk_in_top_k(rag, query, expected):
    hits = rag.retrieve(query, k=3)
    assert any(expected in h["text"] for h in hits)
