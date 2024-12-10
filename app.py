import streamlit as st
from langchain_openai import ChatOpenAI

st.title("💬 T-Chat")

llm = ChatOpenAI(model="gpt-4o")

user_input = st.text_input("You: ")

if st.button("Send"):
    output = llm.invoke(user_input).content
    st.write("Output:", output)