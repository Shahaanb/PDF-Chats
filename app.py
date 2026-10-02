from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory

from rag.chain import build_rag_chain, cited_numbers, with_memory
from rag.ingest import TOP_K, build_vectorstore, load_pdf, load_uploaded_pdfs, make_retriever, split_documents
from rag.providers import (PROVIDERS, available_providers, chat_model_name, embeddings_label, get_chat_model,
                           get_embeddings)
from Templates import bot_avatar, css, user_avatar

load_dotenv()

SAMPLE_DOCS = sorted(Path(__file__).parent.joinpath("sample_docs").glob("*.pdf"))


def init_state():
    st.session_state.setdefault("chain", None)
    st.session_state.setdefault("history", InMemoryChatMessageHistory())
    st.session_state.setdefault("sources", [])  # retrieved chunks + search query for each assistant reply
    st.session_state.setdefault("doc_stats", None)


def clear_chat():
    st.session_state.history = InMemoryChatMessageHistory()
    st.session_state.sources = []


def process_documents(docs, n_files, provider, top_k, search_type):
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks, get_embeddings(provider))
    retriever = make_retriever(vectorstore, top_k, search_type)
    st.session_state.chain = with_memory(build_rag_chain(get_chat_model(provider), retriever),
                                         lambda _session_id: st.session_state.history)
    clear_chat()  # a new document set starts a new conversation
    st.session_state.doc_stats = {"files": n_files, "pages": len(docs), "chunks": len(chunks),
                                  "model": f"{PROVIDERS[provider]['label']} · {chat_model_name(provider)}",
                                  "embeddings": embeddings_label(provider)}


def sidebar():
    with st.sidebar:
        st.subheader("Your Documents")
        files = st.file_uploader("Upload PDFs, then click Process", type="pdf", accept_multiple_files=True)
        providers = available_providers()
        with st.expander("Settings"):
            provider = st.radio("Model provider", providers or ["openai"],
                                format_func=lambda p: PROVIDERS[p]["label"], disabled=len(providers) < 2)
            top_k = st.slider("Passages retrieved per question (k)", 2, 10, TOP_K)
            search_type = st.radio("Search", ["mmr", "similarity"],
                                   format_func={"mmr": "MMR (relevant + diverse)", "similarity": "Similarity"}.get)
        process = st.button("Process", disabled=not files or not providers, width="stretch")
        use_samples = bool(SAMPLE_DOCS) and st.button(f"Use the {len(SAMPLE_DOCS)} sample papers",
                                                      disabled=not providers, width="stretch")
        if process or use_samples:
            with st.spinner("Reading, chunking and embedding the PDFs..."):
                try:
                    if process:
                        process_documents(load_uploaded_pdfs(files), len(files), provider, top_k, search_type)
                    else:
                        docs = [d for path in SAMPLE_DOCS for d in load_pdf(path)]
                        process_documents(docs, len(SAMPLE_DOCS), provider, top_k, search_type)
                except Exception as exc:  # shown to the user instead of a stack trace
                    st.error(f"Could not process the documents: {exc}")
        stats = st.session_state.doc_stats
        if stats:
            st.success(f"Indexed {stats['files']} file(s): {stats['pages']} pages, {stats['chunks']} chunks.\n\n"
                       f"Model: {stats['model']}\n\nEmbeddings: {stats['embeddings']}")
            st.button("Clear chat", on_click=clear_chat, width="stretch")


def show_sources(answer, retrieved):
    """List the retrieved passages under an answer, marking the ones the answer cites."""
    docs = retrieved["docs"]
    if retrieved["query"]:
        st.caption(f"Searched for: *{retrieved['query']}*")
    cited = cited_numbers(answer, len(docs))
    label = f"Sources – {len(cited)} of {len(docs)} passages cited" if cited else f"Sources – {len(docs)} passages"
    with st.expander(label):
        for i, d in enumerate(docs, start=1):
            mark = "cited" if i in cited else "retrieved"
            snippet = " ".join(d.page_content.split())[:350]
            st.markdown(f"**[{i}] {d.metadata.get('source')} – page {d.metadata.get('page')}** "
                        f"<span class='src-tag {mark}'>{mark}</span>", unsafe_allow_html=True)
            st.caption(snippet + ("…" if len(d.page_content) > 350 else ""))


def show_history():
    reply = 0
    for msg in st.session_state.history.messages:
        is_user = msg.type == "human"
        with st.chat_message("user" if is_user else "assistant", avatar=user_avatar if is_user else bot_avatar):
            st.markdown(msg.content)
            if not is_user:
                if reply < len(st.session_state.sources):
                    show_sources(msg.content, st.session_state.sources[reply])
                reply += 1


def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.markdown(css, unsafe_allow_html=True)
    init_state()
    st.header("Chat with PDFs :books:")
    st.caption("Academic research assistant: ask questions about your papers and get answers with page citations.")
    if not available_providers():
        st.error("No API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY in a .env file or as a GitHub Codespaces secret.")
    sidebar()
    show_history()

    if st.session_state.chain is None:
        st.info("Upload one or more PDFs in the sidebar and click **Process** to start chatting.")
    question = st.chat_input("Ask a question about your documents", disabled=st.session_state.chain is None)
    if question:
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(question)
        with st.chat_message("assistant", avatar=bot_avatar):
            with st.spinner("Searching the documents..."):
                try:
                    result = st.session_state.chain.invoke(
                        {"question": question}, config={"configurable": {"session_id": "streamlit"}})
                except Exception as exc:
                    st.error(f"The model call failed: {exc}")
                    return
            st.markdown(result["answer"])
            # Show the rewritten search query when memory turned a follow-up into a standalone question.
            query = result["standalone_question"] if result["standalone_question"] != question else None
            retrieved = {"docs": result["context"], "query": query}
            st.session_state.sources.append(retrieved)
            show_sources(result["answer"], retrieved)


if __name__ == "__main__":
    main()
