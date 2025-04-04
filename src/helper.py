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
    

def setup_pinecone():
    # set up pinecone db
    # Initialize a Pinecone client with  API key
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "dense-index"

    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model":"llama-text-embed-v2",
                "field_map":{"text": "chunk_text"}
            }
        )


    # Connect to the existing Pinecone index
    index = pc.Index(index_name)


def create_embedding_model():
    return OllamaEmbeddings(model='deepseek-r1:7b')
