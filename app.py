import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from Templates import css, bot_template, user_template


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY= os.getenv("GOOGLE_API_KEY")


def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html = True)
    
    st.header("Chat with PDFs")
    user_question = st.text_input("Questions about Document: ")
    if user_question:
        st.write(user_template.replace("{{MSG}}",user_question), unsafe_allow_html = True)
        response = handle_input(user_question)
        st.write(bot_template.replace("{{MSG}}",response), unsafe_allow_html = True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    with st.sidebar:
        st.subheader("Your Documents: ")
        documents = st.file_uploader("Upload Your Files", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing PDF's"):
                # Getting Text From PDF
                text = get_text(documents)
                chunks = get_chunks(text)
                st.write(chunks)
                # Creating Vector Store
                VectorStore = get_vectorstore(chunks)

                st.session_state.conversation = get_conversation_chain(VectorStore)


def get_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    textSplitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = textSplitter.split_text(text)
    return chunks


def get_vectorstore(textchunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=textchunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    #my_llm = ChatOpenAI()
    my_llm = ChatGoogleGenerativeAI(model="gemini-pro")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=my_llm, 
        retriever=vectorstore.as_retriever(), 
        memory=memory
    )
    return conversation_chain


def handle_input(question):
    response = st.session_state.conversation({'question':question})
    return (response)

if __name__ == "__main__":
    main()
