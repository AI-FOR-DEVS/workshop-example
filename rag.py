from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

loader = PyPDFLoader("telekom_faq.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

index_name = "tfaq"

def fill_index():
  PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)

def query_index(query):
  docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)
  return docsearch.similarity_search(query)

if __name__ == "__main__":
  print(query_index("Wie hoch ist der Bereitstellungspreis für die Glasfasertarife?"))