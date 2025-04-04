import streamlit as st
from src.helper import *
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore 
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from src.prompt import *
import os
import pandas as pd

# Load .env file
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

llm = OllamaLLM(model='deepseek-r1:1.5b')
embedding = OllamaEmbeddings(model='deepseek-r1:1.5b')

# Initialize a Pinecone client with  API key
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "dense-index"

# Create index if it not does exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=1536, # 3584 for 7b parameter | 1536 for 1.5b parameter
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled",
        tags={
            "environment": "development"
        }
    )

# Connect to the existing Pinecone index
index = pc.Index(index_name)

# Create vector store 
docsearch = PineconeVectorStore(
    index=index,  
    embedding=embedding
)

# Create prompt template
prompt = PromptTemplate(template=prompt_template, input_variables=["context",'question'])
chain_type_kwargs={'prompt': prompt}

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwags={'k':2}), chain_type_kwargs=chain_type_kwargs)


st.title("Personal RAG Chatbot")
# Create upload file button
uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

if uploaded_file and not uploaded_file.name in os.listdir(pdfs_directory):
    upload_pdf(uploaded_file)
    file = load_pdf(pdfs_directory + uploaded_file.name) # Load document
    
    text_chunks = split_text_chunks(file) # Split document into chunks
    
    full_chunks_list = [t.page_content for t in text_chunks]

    docsearch.add_texts(full_chunks_list) # Add documents to the vector store

# Create table
df = get_uploaded_pdfs()

st.header("Uploaded files", divider="gray")
if len(df) == 0:
    st.text('Nothing uploaded :(')
else:
    # Display uploaded files
    st.table(df)
    if st.button('Clear All Files'):
        clear_pdfs()
        index.delete(delete_all=True)
        st.rerun()

# Create a chat
question = st.chat_input()

if question:
    st.chat_message('user').write(question)
    response = qa.invoke({"query": question})
    st.chat_message("assistant").write(response)
