import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback


def load_openai_api_key():
    
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Unable to retrieve open api key")
    return openai_api_key

def text_processing_from_pdf(pdf):
    pdf = PdfReader(pdf)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
        
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    return knowledge_base

def main():
    
    st.title("📄PDF Summarizer")
    st.write("Created by Mohammed Rishin Poolat")
    st.divider()

    try:
        os.environ["OPENAI_API_KEY"] = load_openai_api_key()
    except ValueError as e:
        st.error(str(e))
        return  

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    if pdf is not None:
        knowledge_base = text_processing_from_pdf(pdf)

        query = "Summarize the content of the uploaded PDF file in approximately 5 to 10 sentences. Focus on capturing the main ideas and key points discussed in the document. Use your own words and ensure clarity and coherence in the summary."

        if query:
            docs = knowledge_base.similarity_search(query)
            open_ai_model = "gpt-3.5-turbo-16k"
            llm = ChatOpenAI(model=open_ai_model, temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)

            st.subheader('Summary Results:')
            st.write(response)
if __name__ == "__main__":
  main()

    
