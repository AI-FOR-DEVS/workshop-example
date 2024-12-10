import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

def get_pinecone_retriever():
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    pc = Pinecone(api_key=pinecone_api_key)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    index_name = "tfaq"  
    index = pc.Index(index_name)
    
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    
    return vectorstore.as_retriever()