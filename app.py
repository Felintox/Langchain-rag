import streamlit as st

st.title("Chatbot Manual CNC")
pergunta = st.text_input("Sua pergunta:")

if pergunta:
    resposta = qa.invoke(pergunta)
    st.write(resposta["result"])