import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

st.title("💬 T-Chat")

if "memory" not in st.session_state:
  st.session_state.memory = ConversationBufferMemory()

llm = ChatOpenAI(model="gpt-4o")

user_input = st.text_input("You: ")

if st.button("Send"):
    st.session_state.memory.chat_memory.add_user_message(user_input)
    complete_prompt = f"{user_input} \n History: {st.session_state.memory.load_memory_variables({})}"
    output = llm.invoke(complete_prompt).content
    st.session_state.memory.chat_memory.add_ai_message(output)
    st.write("Output:", output)