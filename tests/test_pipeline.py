"""Offline tests for the RAG pipeline: no API keys or network needed.

Fake chat models and a spy retriever stand in for the real services, so these tests check
the wiring (page metadata, chunking, citations, memory and query rewriting), not model quality.
"""
from __future__ import annotations

import warnings

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from evaluate import contains_facts
from rag.chain import NOT_FOUND, build_rag_chain, cited_numbers, format_docs, is_not_found, with_memory
from rag.ingest import build_vectorstore, load_pdf, make_retriever, split_documents

warnings.filterwarnings("ignore", category=DeprecationWarning)


def write_pdf(path, pages: list[str]) -> None:
    """Write a minimal PDF with one line of text per page (enough for pypdf to extract)."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content_id = len(objects)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>"
    out, offsets = "%PDF-1.4\n", []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    path.write_bytes(out.encode("latin-1"))


class SpyRetriever(BaseRetriever):
    """Returns fixed documents and remembers every query it was asked."""
    docs: list[Document]
    queries: list[str] = []

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        self.queries.append(query)
        return self.docs


DOCS = [Document(page_content="The encoder has 6 layers.", metadata={"source": "a.pdf", "page": 3}),
        Document(page_content="It uses 8 heads.", metadata={"source": "a.pdf", "page": 5})]


def make_chain(llm, history):
    retriever = SpyRetriever(docs=DOCS, queries=[])
    return with_memory(build_rag_chain(llm, retriever), lambda _sid: history), retriever


CFG = {"configurable": {"session_id": "test"}}


# --- ingestion ----------------------------------------------------------------

def test_load_pdf_keeps_file_name_and_one_based_pages(tmp_path):
    pdf = tmp_path / "paper.pdf"
    write_pdf(pdf, ["First page text", "Second page text"])
    docs = load_pdf(pdf)
    assert [d.metadata for d in docs] == [{"source": "paper.pdf", "page": 1}, {"source": "paper.pdf", "page": 2}]
    assert "Second page" in docs[1].page_content


def test_load_pdf_uses_display_name_for_uploads(tmp_path):
    pdf = tmp_path / "tmp123.pdf"
    write_pdf(pdf, ["Hello"])
    assert load_pdf(pdf, display_name="notes.pdf")[0].metadata["source"] == "notes.pdf"


def test_split_documents_respects_size_and_copies_page_metadata():
    page = Document(page_content=" ".join(f"word{i}" for i in range(600)), metadata={"source": "x.pdf", "page": 7})
    chunks = split_documents([page], chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 500 for c in chunks)
    assert all(c.metadata["page"] == 7 and c.metadata["source"] == "x.pdf" for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
    # consecutive chunks overlap
    assert chunks[0].page_content[-50:].split()[-1] in chunks[1].page_content


def test_build_vectorstore_rejects_empty_input():
    with pytest.raises(ValueError, match="No extractable text"):
        build_vectorstore([], DeterministicFakeEmbedding(size=16))


def test_vectorstore_retrieves_identical_text_first():
    chunks = [Document(page_content=t, metadata={"source": "s.pdf", "page": i})
              for i, t in enumerate(["alpha beta", "gamma delta", "epsilon zeta"], start=1)]
    store = build_vectorstore(chunks, DeterministicFakeEmbedding(size=32))
    assert store.similarity_search("gamma delta", k=1)[0].metadata["page"] == 2


# --- citations ------------------------------------------------------------------

def test_format_docs_numbers_passages_with_source_and_page():
    text = format_docs(DOCS)
    assert text.startswith("[1] (a.pdf, page 3)\nThe encoder has 6 layers.")
    assert "[2] (a.pdf, page 5)" in text


def test_cited_numbers_dedupes_and_ignores_out_of_range():
    assert cited_numbers("Six layers [2][1], see also [2] and [9].", n_passages=4) == [2, 1]
    assert cited_numbers("No citations here.", n_passages=4) == []
    # comma style seen in the evaluation run, e.g. "uses h = 8 heads [1, 2]"
    assert cited_numbers("uses h = 8 heads [1, 2] and d = 64 [3,1].", n_passages=4) == [1, 2, 3]


def test_is_not_found_accepts_curly_apostrophe():
    assert is_not_found(NOT_FOUND)
    assert is_not_found("I couldn’t find this in the uploaded documents.")
    assert not is_not_found("The encoder has 6 layers [1].")


# --- chain and memory -------------------------------------------------------------

def test_first_turn_retrieves_with_the_question_and_skips_rewriting():
    history = InMemoryChatMessageHistory()
    chain, retriever = make_chain(FakeListChatModel(responses=["Six layers [1]."]), history)
    out = chain.invoke({"question": "How many layers?"}, config=CFG)
    assert retriever.queries == ["How many layers?"]
    assert out["standalone_question"] == "How many layers?"
    assert out["answer"] == "Six layers [1]."
    assert out["context"] == DOCS


def test_follow_up_is_rewritten_before_retrieval_and_history_grows():
    history = InMemoryChatMessageHistory()
    llm = FakeListChatModel(responses=["Six layers [1].", "How many heads does the encoder use?", "Eight [2]."])
    chain, retriever = make_chain(llm, history)
    chain.invoke({"question": "How many layers?"}, config=CFG)
    out = chain.invoke({"question": "And heads?"}, config=CFG)
    assert retriever.queries[-1] == "How many heads does the encoder use?"
    assert out["answer"] == "Eight [2]."
    assert [m.type for m in history.messages] == ["human", "ai", "human", "ai"]


def test_answer_prompt_contains_numbered_context_and_history():
    seen = []

    def fake_llm(prompt_value):
        seen.append(prompt_value.to_messages())
        return AIMessage(content="ok")

    history = InMemoryChatMessageHistory()
    chain, _ = make_chain(RunnableLambda(fake_llm), history)
    chain.invoke({"question": "Q1"}, config=CFG)
    system = seen[-1][0].content
    assert "[1] (a.pdf, page 3)" in system and "[2] (a.pdf, page 5)" in system
    assert NOT_FOUND in system
    chain.invoke({"question": "Q2"}, config=CFG)
    # second turn: rewrite call + answer call, and the answer call sees the earlier turn
    assert [m.content for m in seen[-1][1:3]] == ["Q1", "ok"]


def recording_llm(rewrite_to: str, seen: list):
    """Fake model: answers the rewrite prompt with `rewrite_to`, anything else with 'answer'."""
    def respond(prompt_value):
        messages = prompt_value.to_messages()
        seen.append(messages)
        is_rewrite = "rewrite the question" in messages[0].content
        return AIMessage(content=rewrite_to if is_rewrite else "answer")
    return RunnableLambda(respond)


def test_follow_up_answer_step_gets_the_standalone_question():
    seen, history = [], InMemoryChatMessageHistory()
    chain, _ = make_chain(recording_llm("How many heads does the base model use?", seen), history)
    chain.invoke({"question": "What is multi-head attention?"}, config=CFG)
    chain.invoke({"question": "How many of them does it use?"}, config=CFG)
    assert seen[-1][-1].content == "How many heads does the base model use?"
    # memory still stores what the user actually typed
    assert history.messages[2].content == "How many of them does it use?"


def test_baseline_answer_step_gets_the_raw_follow_up():
    seen, history = [], InMemoryChatMessageHistory()
    retriever = SpyRetriever(docs=DOCS, queries=[])
    chain = with_memory(build_rag_chain(recording_llm("Rewritten?", seen), retriever, answer_standalone=False),
                        lambda _sid: history)
    chain.invoke({"question": "First?"}, config=CFG)
    chain.invoke({"question": "And them?"}, config=CFG)
    assert retriever.queries[-1] == "Rewritten?"
    assert seen[-1][-1].content == "And them?"


def test_make_retriever_configures_mmr_and_similarity():
    store = build_vectorstore([Document(page_content=f"text {i}", metadata={"source": "s.pdf", "page": i})
                               for i in range(12)], DeterministicFakeEmbedding(size=16))
    mmr = make_retriever(store, k=6)
    assert mmr.search_type == "mmr" and mmr.search_kwargs == {"k": 6, "fetch_k": 30, "lambda_mult": 0.5}
    assert len(mmr.invoke("text 3")) == 6
    sim = make_retriever(store, k=4, search_type="similarity")
    assert sim.search_type == "similarity" and sim.search_kwargs == {"k": 4}


# --- evaluation helpers -------------------------------------------------------------

def test_contains_facts_needs_every_group():
    assert contains_facts("BERT masks 15% of tokens", [["15%", "15 percent"]])
    assert contains_facts("Masked LM and Next Sentence Prediction", [["mask"], ["next sentence", "nsp"]])
    assert not contains_facts("Masked LM only", [["mask"], ["next sentence", "nsp"]])
