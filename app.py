import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from rag import query_index
st.title("💬 T-Chat")

if "memory" not in st.session_state:
  st.session_state.memory = ConversationBufferMemory()

llm = ChatOpenAI(model="gpt-4o")

user_input = st.text_input("You: ")

if st.button("Send"):
    matching_docs = query_index(user_input)
    st.session_state.memory.chat_memory.add_user_message(user_input)
    complete_prompt = f"{user_input} \n History: {st.session_state.memory.load_memory_variables({})} \n Nutze folgende Informationen als Kontext wenn du nicht weiterkommst: {matching_docs}"
    output = llm.invoke(complete_prompt).content
    st.session_state.memory.chat_memory.add_ai_message(output)
    st.write("Output:", output)