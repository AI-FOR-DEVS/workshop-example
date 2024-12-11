from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
client = QdrantClient(":memory:")

client.create_collection(
    collection_name="tfaq",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

loader = TextLoader("llama3.txt")
pdf = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

chunks = text_splitter.split_documents(pdf)


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="tfaq",
    embedding=embeddings,
)

def fill_index():
  vector_store.add_documents(chunks)

def query_index(query):
  return vector_store.similarity_search(query)

if __name__ == "__main__":
  fill_index()
  print(query_index("Is it possible to use Llama 3.1 for slack?"))