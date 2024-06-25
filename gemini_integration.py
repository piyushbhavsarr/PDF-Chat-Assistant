import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
import os
import google.generativeai as genai
import subprocess
import spacy
from spacy.cli import download


# Ensure the Spacy model is downloaded and loaded
model_name = "en_core_web_sm"
try:
    nlp = spacy.load(model_name)
except OSError:
    download(model_name)
    nlp = spacy.load(model_name)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Spacy embeddings
embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# Gemini model initialization
model = genai.GenerativeModel('gemini-1.5-flash')
chat = model.start_chat(history=[])

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

def get_conversational_chain(retrieval_tool, user_question):
    # Add system message to the chat history
    system_message = """You are a helpful assistant. Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in the provided context, just say, "answer is not available in the context", don't provide the wrong answer."""
    chat.history.append({'role': 'model', 'parts': [system_message]})
    
    # Add the user's question to the chat history
    chat.history.append({'role': 'user', 'parts': [user_question]})
    
    # Retrieve relevant context from PDF using retrieval tool
    pdf_context = ""
    if retrieval_tool:
        tool_response = retrieval_tool.run(user_question)
        if isinstance(tool_response, str):
            pdf_context = tool_response
        else:
            pdf_context = str(tool_response)
    
    if pdf_context:
        # Add the PDF context to the chat history
        chat.history.append({'role': 'model', 'parts': [pdf_context]})
        
    # Send the questions to the Gemini model
    response = model.generate_content(chat.history)
    
    # Extract the response text from the first candidate
    response_text = response.candidates[0].content.parts[0].text
    
    # Display the model's response
    st.write("Reply: ", response_text)

def user_input(user_question):
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    retrieval_tool = create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answer to queries from the pdf")
    get_conversational_chain(retrieval_tool, user_question)

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("RAG based Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
