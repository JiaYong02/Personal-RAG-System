from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore 
from src.helper import load_pdf, split_text_chunks, create_embedding_model

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Initialize a Pinecone client with  API key
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "dense-index"

# Create index if it not does exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=3584,
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
    embedding=create_embedding_model()
)