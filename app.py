from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from retrievers import get_pinecone_retriever
from langchain.chains.retrieval import create_retrieval_chain

def get_chain():
  llm = ChatOpenAI(model="gpt-4o", temperature=0)

  prompt = ChatPromptTemplate.from_messages([
      ("system", "Du bist ein deutsche Mitarbeiter der T-Firma. Antworte auf die letzte Frage des Users in einem Satz basierend auf folgenden Kontext: {context}"),
      ("user", "{input}"),
      MessagesPlaceholder(variable_name="chat_history")
  ])

  # Print the prompt template
  print("--------------------\nPrompt Template:")
  print(prompt.format(context="<context>", input="<input>", chat_history=[]))


  chain = create_stuff_documents_chain(llm, prompt)

  retriever = get_pinecone_retriever()
  return create_retrieval_chain(retriever, chain)

def stream_response(query, chat_history):
  chain = get_chain()
  for chunk in chain.stream({"input": query, "chat_history": chat_history}):
    if "answer" in chunk:
      yield chunk["answer"]

if __name__ == "__main__":
  for chunk in stream_response("Hi", []):
      print(chunk, end='', flush=True)