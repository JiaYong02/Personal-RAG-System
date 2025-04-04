from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings


# Load data from pdf file
def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()
    


# Split text into chunks
def split_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        add_start_index=True
    )
    # get text chunks
    return text_splitter.split_documents(documents)



def create_embedding_model():
    return OllamaEmbeddings(model='deepseek-r1:7b')
