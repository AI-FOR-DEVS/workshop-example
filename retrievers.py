import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

def get_pinecone_retriever():
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    pc = Pinecone(api_key=pinecone_api_key)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    index_name = "tfaq"  
    index = pc.Index(index_name)
    
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    
    return vectorstore.as_retriever()

def get_qdrant_retriever():
    client = QdrantClient(":memory:")
    
    client.create_collection(
        collection_name="tfaq",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="tfaq",
        embedding=embeddings,
    )

    loader = PyPDFLoader("telekom_faq.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    vectorstore.add_documents(chunks)

    return vectorstore.as_retriever()