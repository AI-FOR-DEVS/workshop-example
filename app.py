from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

print(llm.invoke("Tell me a joke"))