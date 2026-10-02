"""Document ingestion: PDF loading, chunking and the FAISS vector store."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 6


def load_pdf(path: str | Path, display_name: str | None = None) -> list[Document]:
    """Load one PDF as page-level Documents.

    Each Document carries the file name (`source`) and the 1-based page number
    (`page`) so answers can be traced back to where the text came from.
    Pages without extractable text (blank or scanned pages) are skipped.
    """
    name = display_name or Path(path).name
    docs = []
    for page in PyPDFLoader(str(path)).load():
        text = page.page_content.strip()
        if not text:
            continue
        docs.append(Document(page_content=text,
                             metadata={"source": name, "page": page.metadata.get("page", 0) + 1}))
    return docs


def load_uploaded_pdfs(files) -> list[Document]:
    """Load Streamlit UploadedFile objects. PyPDFLoader needs a path, so each upload goes through a temp file."""
    docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        try:
            docs.extend(load_pdf(tmp_path, display_name=f.name))
        finally:
            os.remove(tmp_path)
    return docs


def split_documents(docs: list[Document], chunk_size: int = CHUNK_SIZE,
                    chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    """Split pages into overlapping chunks; the page metadata is copied onto every chunk."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              add_start_index=True)
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def build_vectorstore(chunks: list[Document], embeddings: Embeddings) -> FAISS:
    if not chunks:
        raise ValueError("No extractable text was found in the uploaded PDFs (are they scanned images?).")
    return FAISS.from_documents(chunks, embeddings)


def make_retriever(vectorstore: FAISS, k: int = TOP_K, search_type: str = "mmr") -> VectorStoreRetriever:
    """Retriever over the index.

    MMR (maximal marginal relevance) first fetches the 30 closest chunks and then picks k that
    are relevant but not near-duplicates of each other. In our evaluation this pulled the right
    passage into the top 6 for questions where plain similarity search returned several
    overlapping chunks from one page or one paper (see eval/results).
    """
    search_kwargs = {"k": k}
    if search_type == "mmr":
        search_kwargs.update(fetch_k=max(30, 4 * k), lambda_mult=0.5)
    return vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
