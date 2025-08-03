import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon = ":books:")
    st.header("Chat with PDFs")
    st.text_input("Questions about Document: ")

    with st.sidebar:
        st.subheader("Your Documents: ")
        documents = st.file_uploader("Upload Your Files", accept_multiple_files= True)
        if st.button("Process"):
            with st.spinner("Processing PDF's"):
                #Getting Text From PDF
                text = get_text(documents)
                chunks = get_chunks(text)
                st.write(chunks)
                #Creating Vector Store
        

def get_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    textSplitter = CharacterTextSplitter( separator = "\n", chunk_size = 1000, chunk_overlap = 200, length_function = len)
    chunks = textSplitter.split_text(text)
    return chunks


if __name__ == "__main__":
    main()