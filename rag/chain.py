"""The conversational RAG chain, composed with LCEL.

    question + chat history
        -> standalone question   (ChatPromptTemplate | LLM | StrOutputParser, skipped on the first turn)
        -> retrieved chunks      (VectorStoreRetriever)
        -> grounded answer       (ChatPromptTemplate | LLM | StrOutputParser)

The chain returns a dict with the standalone question, the retrieved
Documents and the answer, and is wrapped in RunnableWithMessageHistory so
follow-up questions can refer back to earlier turns.
"""
from __future__ import annotations

from operator import itemgetter
from typing import Callable

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableBranch, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

# Only the most recent turns are sent to the model, which keeps prompts small in long chats.
HISTORY_MESSAGES = 10

CONDENSE_SYSTEM = (
    "Given the chat history and the latest user question, rewrite the question so that it can be "
    "understood without the chat history. Resolve references such as 'it', 'they' or 'that model'. "
    "Do NOT answer the question. Return only the rewritten question, or the question unchanged if it "
    "is already standalone."
)

ANSWER_SYSTEM = (
    "You are an academic research assistant. Answer the user's question using only the context "
    "passages taken from their uploaded PDFs.\n"
    "If the context does not contain the answer, say that you could not find it in the uploaded "
    "documents instead of using outside knowledge.\n"
    "Keep numbers, names and terminology exactly as they appear in the context.\n\n"
    "Context:\n{context}"
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"({d.metadata.get('source')}, page {d.metadata.get('page')})\n{d.page_content}" for d in docs
    )


def build_rag_chain(llm: BaseChatModel, retriever: BaseRetriever) -> Runnable:
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system", CONDENSE_SYSTEM),
        MessagesPlaceholder("chat_history", n_messages=HISTORY_MESSAGES),
        ("human", "{question}"),
    ])
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_SYSTEM),
        MessagesPlaceholder("chat_history", n_messages=HISTORY_MESSAGES),
        ("human", "{question}"),
    ])

    # On the first turn there is nothing to resolve, so skip the extra LLM call.
    standalone_question = RunnableBranch(
        (lambda x: not x.get("chat_history"), itemgetter("question")),
        condense_prompt | llm | StrOutputParser(),
    )
    answer = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | answer_prompt
        | llm
        | StrOutputParser()
    )
    return (
        RunnablePassthrough.assign(standalone_question=standalone_question)
        | RunnablePassthrough.assign(context=itemgetter("standalone_question") | retriever)
        | RunnablePassthrough.assign(answer=answer)
    )


def with_memory(chain: Runnable, get_history: Callable[[str], BaseChatMessageHistory]) -> Runnable:
    """Store each question and answer in the session's chat history and feed it back on the next turn."""
    return RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
