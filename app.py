import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory

from rag.chain import build_rag_chain, with_memory
from rag.ingest import build_vectorstore, load_uploaded_pdfs, split_documents
from rag.providers import PROVIDERS, chat_model_name, default_provider, get_chat_model, get_embeddings
from Templates import bot_avatar, user_avatar

load_dotenv()

TOP_K = 4


def init_state():
    st.session_state.setdefault("chain", None)
    st.session_state.setdefault("history", InMemoryChatMessageHistory())
    st.session_state.setdefault("doc_stats", None)


def process_documents(files):
    provider = default_provider()
    docs = load_uploaded_pdfs(files)
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks, get_embeddings(provider))
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    st.session_state.chain = with_memory(build_rag_chain(get_chat_model(provider), retriever),
                                         lambda _session_id: st.session_state.history)
    # A new document set starts a new conversation.
    st.session_state.history = InMemoryChatMessageHistory()
    st.session_state.doc_stats = {"files": len(files), "pages": len(docs), "chunks": len(chunks),
                                  "model": f"{PROVIDERS[provider]['label']} · {chat_model_name(provider)}"}


def sidebar():
    with st.sidebar:
        st.subheader("Your Documents")
        files = st.file_uploader("Upload PDFs, then click Process", type="pdf", accept_multiple_files=True)
        if st.button("Process", disabled=not files or default_provider() is None, use_container_width=True):
            with st.spinner("Reading, chunking and embedding the PDFs..."):
                try:
                    process_documents(files)
                except Exception as exc:  # shown to the user instead of a stack trace
                    st.error(f"Could not process the documents: {exc}")
        stats = st.session_state.doc_stats
        if stats:
            st.success(f"Indexed {stats['files']} file(s): {stats['pages']} pages, {stats['chunks']} chunks.\n\n"
                       f"Model: {stats['model']}")


def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    init_state()
    st.header("Chat with PDFs :books:")
    if default_provider() is None:
        st.error("No API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY in a .env file or as a GitHub Codespaces secret.")
    sidebar()

    for msg in st.session_state.history.messages:
        is_user = msg.type == "human"
        with st.chat_message("user" if is_user else "assistant", avatar=user_avatar if is_user else bot_avatar):
            st.markdown(msg.content)

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
                    st.markdown(result["answer"])
                except Exception as exc:
                    st.error(f"The model call failed: {exc}")


if __name__ == "__main__":
    main()
