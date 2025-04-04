from langchain_community.document_loaders import PyPDFDirectoryLoader, PDFPlumberLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import os 
import pandas as pd


pdfs_directory = 'pdfs/'

def upload_pdf(file):
    # Ensure the directory exists
    os.makedirs(pdfs_directory, exist_ok=True)

    # Save the file
    file_path = os.path.join(pdfs_directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())


# Load data from pdf file
def load_pdf(path):
    loader = PDFPlumberLoader(path)
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

# Remove all file
def clear_pdfs():
    for file_name in os.listdir(pdfs_directory):
        file_path = os.path.join(pdfs_directory, file_name)
        if os.path.isfile(file_path) and file_name != ".gitkeep":
            os.remove(file_path)


def get_uploaded_pdfs():
    file_list = [os.path.splitext(file)[0] for file in os.listdir(pdfs_directory) if file != '.gitkeep'] 
    df = pd.DataFrame(file_list, columns=['File name'])
    df.index = range(1, len(df) + 1)
    return df