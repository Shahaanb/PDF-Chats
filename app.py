import streamlit as st
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon = ":books:")
    st.header("Chat with PDFs")
    st.text_input("Questions about Document: ")

    with st.sidebar:
        st.subheader("Your Documents: ")
        st.file_uploader("Upload Your Files")
        st.button("Process")
        

if __name__ == "__main__":
    main()