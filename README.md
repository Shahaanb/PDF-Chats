# 📄 Chat With PDFs

A **Streamlit-based web application** that enables users to have intelligent conversations with PDF documents. Upload one or more PDFs and ask questions — the app will retrieve relevant content and respond with natural language answers powered by **LLMs** (Large Language Models).

---

## 🧠 What It Does

This project allows you to:

- Upload PDF files
- Automatically extract and process text
- Ask context-aware questions in natural language
- Get conversational responses based on your documents
- Maintain chat history for follow-up questions

---

## 🚀 Features

- **Multiple PDF Uploads:** Supports uploading and querying multiple PDFs at once.
- **Text Extraction:** Uses `PyPDF2` to read and extract text from documents.
- **Retrieval Augmented Generation (RAG):** Combines retrieval-based search with generative models.
- **Conversational Memory:** Remembers chat history for context in follow-ups.
- **Streamlit UI:** Clean, interactive, and responsive user interface.
- **LLM Support:** Easily switch between OpenAI, Hugging Face, and Google Generative AI.

---

## ⚙️ How It Works (RAG Pipeline)

1. **PDF Upload:** Upload one or more PDFs through the Streamlit sidebar.
2. **Text Extraction:** Extract text content from each page.
3. **Text Chunking:** Break long documents into manageable text chunks.
4. **Embeddings:** Convert chunks into numerical vectors using a SentenceTransformer (e.g. OpenAI embeddings).
5. **Vector Store:** Store embeddings in a **FAISS** index for fast similarity search.
6. **Conversational Retrieval:**
   - Embed user’s question
   - Retrieve relevant chunks using FAISS
   - Pass both to an LLM (like GPT-4)
   - Return the LLM’s answer
   - Maintain history for context

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Make sure you have a `requirements.txt` with at least:

```
streamlit
langchain
openai
faiss-cpu
PyPDF2
python-dotenv
langchain-google-genai
langchain-community
```

Install with:

```bash
pip install -r requirements.txt
```

### 4. Add API Keys

Create a `.env` file in the root directory with:

```dotenv
OPENAI_API_KEY="your_openai_api_key"
HUGGINGFACEHUB_API_TOKEN="your_huggingface_api_token"
GOOGLE_API_KEY="your_google_api_key"
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Once the app launches:

1. Upload PDFs from the sidebar.
2. Click "Process" to load and chunk the documents.
3. Ask questions in the input box.
4. Read answers in the chat window.

---

## 🧩 Current LLM Support

- **OpenAI GPT (default)**

---

## 🛣️ Future Improvements

- 🔄 Add support for `.docx`, `.csv`, `.txt` files
- 🧩 Enable full support for Hugging Face and Gemini by default
- 📚 Display source document snippets with responses
- 🧹 Add “Clear Chat” button
- ⚡ Performance optimization for large documents
- 👍 User feedback system for improving answer quality

---

## 📌 Summary

> “Chat With PDFs” makes document reading interactive and efficient. Whether it’s lengthy research papers, contracts, or textbooks — get your answers in seconds using the power of LLMs.
