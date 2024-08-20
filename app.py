import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from huggingface_hub import login
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback


def load_huggingface_api_token():
    load_dotenv()
    huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    print(huggingface_token)
    if not huggingface_token:
        raise ValueError("Unable to retrieve open ai api key")
    return huggingface_token

def text_processing_from_pdf(pdf):
    
    pdf = PdfReader(pdf)
    text = ""
    
    for page in pdf.pages:
        text += page.extract_text()
        
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    # ... existing code ...
    
    return knowledge_base

def main():
    
    st.title("📄PDF Summarizer")
    st.write("Created by Mohammed Rishin Poolat")
    st.divider()

    try:
        huggingface_token = load_huggingface_api_token()
        login(token=huggingface_token)
    except ValueError as e:
        st.error(str(e))
        return  

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    if pdf is not None:
        knowledge_base = text_processing_from_pdf(pdf)

        query = "Summarize the content of the uploaded PDF file in approximately 5 to 10 sentences. Focus on capturing the main ideas and key points discussed in the document. Use your own words and ensure clarity and coherence in the summary."

        if query:
            docs = knowledge_base.similarity_search(query)
            llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5, "max_length": 512})
            chain = load_qa_chain(llm, chain_type='stuff')

            response = chain.run(input_documents=docs, question=query)


            st.subheader('Summary Results:')
            st.write(response)
            
if __name__ == "__main__":
  main()

    
