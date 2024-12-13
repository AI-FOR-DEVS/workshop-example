from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from retrievers import get_qdrant_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.tracers.context import tracing_v2_enabled

def get_chain():
  llm = ChatOpenAI(model="gpt-4o", temperature=0)

  prompt = ChatPromptTemplate.from_messages([
      ("system", """
        Du bist ein deutsche Mitarbeiter der T-Firma. 
        Antworte auf die letzte Frage des Users in einem Satz basierend auf folgenden Kontext: {context}

       """),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}")
  ])

  chain = create_stuff_documents_chain(llm, prompt)

  retriever = get_qdrant_retriever()
  return create_retrieval_chain(retriever, chain)

def stream_response(query, chat_history):
  chain = get_chain()
  for chunk in chain.stream({"input": query, "chat_history": chat_history}):
    if "answer" in chunk:
      yield chunk["answer"]

if __name__ == "__main__":
  for chunk in stream_response("Hi", []):
      print(chunk, end='', flush=True)