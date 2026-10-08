import pytest
from rag import chunk_text


def test_overlap_and_coverage():
    text = "word " * 1000
    chunks = chunk_text(text, size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_short_text_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_bad_params():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)
