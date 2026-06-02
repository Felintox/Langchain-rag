import streamlit as st
import requests

st.title("Interface Cliente - RAG")
st.caption("Conectado à API FastAPI")

pergunta = st.text_input("Faça sua pergunta sobre ARC-MOF:")

if pergunta:
    with st.spinner("Consultando a API..."):
        try:
            # Faz a requisição HTTP POST para o nosso main.py
            resposta = requests.post(
                "http://127.0.0.1:8000/pergunta", 
                json={"pergunta": pergunta}
            )
            
            if resposta.status_code == 200:
                dados = resposta.json()
                st.write(dados.get("resposta"))
            else:
                st.error(f"Erro na API: {resposta.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Erro: Não foi possível conectar à API. Certifique-se de que o main.py está rodando com uvicorn.")
