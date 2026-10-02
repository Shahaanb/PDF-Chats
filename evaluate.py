"""Evaluate the RAG assistant on the test questions in eval/questions.json.

For every question it records the retrieved pages, the answer and four checks:

* retrieval hit  - one of the expected (file, page) sources is among the k retrieved chunks
* correct        - the answer contains the expected key facts (or, for out-of-scope
                   questions, says the documents do not contain the answer)
* cited          - the answer cites at least one retrieved passage as [n]
* grounded       - an LLM judge compares the answer with the retrieved passages; its verdict
                   is parsed into a Pydantic model with PydanticOutputParser

Questions that share a "conversation" id are asked in order in one chat, so follow-ups
exercise the memory and the query-rewriting step.

    python sample_docs/download_samples.py
    python evaluate.py                          # final configuration -> eval/results/
    python evaluate.py --baseline --out eval/results_baseline
                                                # first version: similarity search, k = 4,
                                                # answer step sees the raw follow-up question
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag.chain import build_rag_chain, cited_numbers, format_docs, is_not_found, with_memory
from rag.ingest import CHUNK_OVERLAP, CHUNK_SIZE, TOP_K, build_vectorstore, load_pdf, make_retriever, split_documents
from rag.providers import chat_model_name, default_provider, embeddings_label, get_chat_model, get_embeddings

warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).parent


class GroundingVerdict(BaseModel):
    verdict: Literal["grounded", "partially grounded", "not grounded"] = Field(
        description="Whether every claim in the answer is supported by the source passages")
    unsupported_claims: list[str] = Field(default_factory=list,
                                          description="Claims in the answer that the passages do not support")
    explanation: str = Field(description="One or two sentences explaining the verdict")


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You check whether an assistant's answer is supported by the source passages it was given. "
     "A claim counts as supported only if the passages state it or it follows directly from them. "
     "Ignore citation markers such as [1]. An answer that says the documents do not contain the "
     "information is grounded when the passages really do not contain it.\n\n{format_instructions}"),
    ("human", "Question: {question}\n\nSource passages:\n{context}\n\nAnswer to check:\n{answer}"),
])


def normalise(text: str) -> str:
    return " ".join(text.lower().replace("’", "'").split())


def contains_facts(answer: str, must_contain: list[list[str]]) -> bool:
    """Every group needs at least one of its alternatives in the answer."""
    a = normalise(answer)
    return all(any(normalise(alt) in a for alt in group) for group in must_contain)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=default_provider())
    ap.add_argument("--k", type=int, default=None, help=f"passages per question (default {TOP_K}, baseline 4)")
    ap.add_argument("--search", choices=["mmr", "similarity"], default=None)
    ap.add_argument("--baseline", action="store_true",
                    help="first version: similarity search, k = 4, raw follow-up in the answer prompt")
    ap.add_argument("--judge-model", default=None,
                    help="model for the grounding judge (default: gemini-3.7-flash, falling back to the chat model)")
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.json"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--pause", type=float, default=4.0, help="seconds between questions (free-tier rate limits)")
    args = ap.parse_args()
    if not args.provider:
        raise SystemExit("No API key found: set GOOGLE_API_KEY or OPENAI_API_KEY.")
    k = args.k or (4 if args.baseline else TOP_K)
    search = args.search or ("similarity" if args.baseline else "mmr")

    pdfs = sorted((ROOT / "sample_docs").glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No sample PDFs found: run python sample_docs/download_samples.py first.")

    t0 = time.time()
    pages = [d for path in pdfs for d in load_pdf(path)]
    chunks = split_documents(pages)
    vectorstore = build_vectorstore(chunks, get_embeddings(args.provider))
    index_seconds = time.time() - t0
    print(f"Indexed {len(pdfs)} PDFs: {len(pages)} pages, {len(chunks)} chunks in {index_seconds:.1f}s")

    llm = get_chat_model(args.provider)
    histories: dict[str, InMemoryChatMessageHistory] = {}
    chain = with_memory(build_rag_chain(llm, make_retriever(vectorstore, k, search),
                                        answer_standalone=not args.baseline),
                        lambda session_id: histories.setdefault(session_id, InMemoryChatMessageHistory()))

    # The judge uses a stronger model than the assistant when one is reachable; the lighter chat
    # model is the fallback when the stronger one is overloaded or its output does not parse.
    parser = PydanticOutputParser(pydantic_object=GroundingVerdict)
    judge_prompt = JUDGE_PROMPT.partial(format_instructions=parser.get_format_instructions())
    judge_model = args.judge_model or ("gemini-3.7-flash" if args.provider == "google" else chat_model_name(args.provider))
    judge = (judge_prompt | get_chat_model(args.provider, model=judge_model, max_retries=1, timeout=60) | parser).with_fallbacks(
        [judge_prompt | llm | parser])

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    results = []
    for q in questions:
        session = q.get("conversation", q["id"])
        start = time.time()
        out = chain.invoke({"question": q["question"]}, config={"configurable": {"session_id": session}})
        latency = time.time() - start
        docs = out["context"]
        retrieved = [[d.metadata["source"], d.metadata["page"]] for d in docs]
        expected = [list(s) for s in q["expected_sources"]]
        hit_ranks = [i for i, r in enumerate(retrieved, start=1) if r in expected]
        not_found = is_not_found(out["answer"])
        if q.get("expect_not_found"):
            correct = not_found
        else:
            correct = not not_found and contains_facts(out["answer"], q["must_contain"])
        cited = cited_numbers(out["answer"], len(docs))
        try:
            verdict = judge.invoke({"question": out["standalone_question"], "context": format_docs(docs),
                                    "answer": out["answer"]})
            grounding = verdict.model_dump()
        except Exception as exc:  # judge output that does not parse is recorded, not fatal
            grounding = {"verdict": "judge error", "unsupported_claims": [], "explanation": str(exc)[:200]}
        row = {
            "id": q["id"], "category": q["category"], "question": q["question"],
            "conversation": q.get("conversation"), "standalone_question": out["standalone_question"],
            "expected_answer": q["expected_answer"], "answer": out["answer"],
            "retrieved": retrieved, "expected_sources": expected,
            "retrieval_hit": bool(hit_ranks) if expected else None,
            "first_hit_rank": hit_ranks[0] if hit_ranks else None,
            "correct": correct, "said_not_found": not_found,
            "cited_passages": cited, "has_citation": bool(cited),
            "grounding": grounding, "latency_s": round(latency, 2),
        }
        results.append(row)
        print(f"{q['id']:>4} correct={correct!s:5} hit={row['retrieval_hit']!s:5} cited={bool(cited)!s:5} "
              f"grounded={grounding['verdict']:<18} {latency:5.1f}s  {q['question']}")
        time.sleep(args.pause)

    meta = {
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "provider": args.provider, "chat_model": chat_model_name(args.provider),
        "embeddings": embeddings_label(args.provider), "k": k, "search": search,
        "answer_step": "raw question" if args.baseline else "standalone question",
        "judge_model": judge_model,
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
        "documents": [p.name for p in pdfs], "pages": len(pages), "chunks": len(chunks),
        "index_seconds": round(index_seconds, 1),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2,
                                                     ensure_ascii=False), encoding="utf-8")
    (out_dir / "results.md").write_text(render_markdown(meta, results), encoding="utf-8")
    print(f"\nWrote {out_dir / 'results.md'}")
    print(summary_line(results))


def summary_line(results: list[dict]) -> str:
    n = len(results)
    with_sources = [r for r in results if r["retrieval_hit"] is not None]
    answerable = [r for r in results if r["expected_sources"]]
    return (f"correct {sum(r['correct'] for r in results)}/{n} | "
            f"retrieval hit@k {sum(r['retrieval_hit'] for r in with_sources)}/{len(with_sources)} | "
            f"cited {sum(r['has_citation'] for r in answerable)}/{len(answerable)} | "
            f"grounded {sum(r['grounding']['verdict'] == 'grounded' for r in results)}/{n} | "
            f"median latency {statistics.median(r['latency_s'] for r in results):.1f}s")


def yes_no(value) -> str:
    return "n/a" if value is None else ("yes" if value else "**no**")


def render_markdown(meta: dict, results: list[dict]) -> str:
    lines = [
        "# Evaluation results", "",
        f"Run {meta['run_at']} · {meta['provider']} `{meta['chat_model']}` · embeddings `{meta['embeddings']}` "
        f"· chunks {meta['chunk_size']}/{meta['chunk_overlap']}", "",
        f"Retrieval: {meta['search']}, k = {meta['k']} · answer step sees the {meta['answer_step']} "
        f"· grounding judge `{meta['judge_model']}` (falls back to the chat model)", "",
        f"Documents: {', '.join(meta['documents'])} ({meta['pages']} pages, {meta['chunks']} chunks, "
        f"indexed in {meta['index_seconds']} s)", "",
        f"**Summary:** {summary_line(results)}", "",
        "| ID | Category | Question | Retrieval hit (rank) | Correct | Cited | Grounded (judge) | Latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        hit = yes_no(r["retrieval_hit"])
        if r["first_hit_rank"]:
            hit += f" ({r['first_hit_rank']})"
        lines.append(f"| {r['id']} | {r['category']} | {r['question']} | {hit} | {yes_no(r['correct'])} | "
                     f"{yes_no(r['has_citation']) if r['expected_sources'] else 'n/a'} | "
                     f"{r['grounding']['verdict']} | {r['latency_s']} s |")
    lines += ["", "## Answers", ""]
    for r in results:
        lines += [f"### {r['id']}. {r['question']}", ""]
        if r["standalone_question"] != r["question"]:
            lines += [f"*Rewritten search query:* {r['standalone_question']}", ""]
        lines += [f"*Expected:* {r['expected_answer']}", "",
                  "*Answer:*", "", "> " + r["answer"].strip().replace("\n", "\n> "), "",
                  "*Retrieved:* " + ", ".join(f"[{i}] {s} p.{p}" for i, (s, p) in enumerate(r["retrieved"], 1)),
                  "", f"*Judge:* {r['grounding']['verdict']} – {r['grounding']['explanation']}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
