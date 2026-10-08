import streamlit as st

from rag import RAG

st.set_page_config(page_title="Chat with your docs", page_icon="📚")
st.title("📚 Chat with your documents")


@st.cache_resource
def get_rag() -> RAG:
    return RAG()


rag = get_rag()

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"],
                             accept_multiple_files=True)
    if st.button("Index files") and files:
        for f in files:
            n = rag.ingest(f.name, f.getvalue())
            st.success(f"{f.name}: {n} chunks")
    k = st.slider("Chunks retrieved (k)", 1, 10, 4)
    st.caption(f"Indexed chunks: {rag.col.count()}")
    if st.button("Clear index"):
        rag.reset()
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

if q := st.chat_input("Ask about your documents"):
    st.chat_message("user").write(q)
    with st.spinner("Thinking..."):
        ans, hits = rag.answer(q, k)
    with st.chat_message("assistant"):
        st.write(ans)
        with st.expander("Sources"):
            for h in hits:
                st.markdown(f"**{h['source']} p.{h['page']}** "
                            f"(distance {h['distance']:.2f})")
                st.caption(h["text"][:300] + "...")
    st.session_state.history += [("user", q), ("assistant", ans)]
