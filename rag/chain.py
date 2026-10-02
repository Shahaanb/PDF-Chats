"""The conversational RAG chain, composed with LCEL.

    question + chat history
        -> standalone question   (ChatPromptTemplate | LLM | StrOutputParser, skipped on the first turn)
        -> retrieved chunks      (VectorStoreRetriever, MMR)
        -> grounded answer       (ChatPromptTemplate | LLM | StrOutputParser) to the standalone
                                 question, citing passages as [n]

The chain returns a dict with the standalone question, the retrieved
Documents and the answer, and is wrapped in RunnableWithMessageHistory so
follow-up questions can refer back to earlier turns.
"""
from __future__ import annotations

import re
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

NOT_FOUND = "I couldn't find this in the uploaded documents."

ANSWER_SYSTEM = (
    "You are an academic research assistant. Answer the user's question using only the numbered "
    "context passages below, which come from the user's uploaded PDFs.\n"
    "Rules:\n"
    "1. Cite the passages that support each statement with their numbers in square brackets, "
    "e.g. [1] or [2][3].\n"
    f"2. If the passages do not contain the answer, reply exactly: \"{NOT_FOUND}\" "
    "Do not use outside knowledge.\n"
    "3. Keep numbers, names and terminology exactly as written in the passages.\n\n"
    "Context:\n{context}"
)

# Matches [1], [2][3] and the comma style the model sometimes uses, [1, 2].
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def format_docs(docs: list[Document]) -> str:
    """Number the passages so the model can cite them as [1], [2], ..."""
    return "\n\n".join(
        f"[{i}] ({d.metadata.get('source')}, page {d.metadata.get('page')})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def cited_numbers(answer: str, n_passages: int) -> list[int]:
    """Passage numbers cited in an answer, in order of first use, ignoring out-of-range numbers."""
    seen = []
    for m in CITATION.finditer(answer):
        for part in m.group(1).split(","):
            n = int(part)
            if 1 <= n <= n_passages and n not in seen:
                seen.append(n)
    return seen


def is_not_found(answer: str) -> bool:
    """True if the model gave the agreed 'not in the documents' reply (curly apostrophes allowed)."""
    return NOT_FOUND.lower().rstrip(".") in answer.lower().replace("’", "'")


def build_rag_chain(llm: BaseChatModel, retriever: BaseRetriever, answer_standalone: bool = True) -> Runnable:
    """answer_standalone=False reproduces our first version, which answered the raw follow-up."""
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
    # The answer step gets the rewritten question too: with a vague follow-up such as
    # "How many of them...?" the model otherwise often replied that it could not find the answer
    # even though the right passage had been retrieved.
    answer = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["context"]),
            question=itemgetter("standalone_question" if answer_standalone else "question"))
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
