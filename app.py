# app.py
import streamlit as st
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate

# ── carrega UMA vez, não reconstrói ──────────────────────────
@st.cache_resource
def carregar_pipeline():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    db = Chroma(
        persist_directory="./db",
        embedding_function=embeddings
    )
    
    llm = Ollama(model="llama3.1:8b")
    
    template = PromptTemplate.from_template("""
    Answer the question based on the context below.
    If you don't know, say you don't know.
    
    Context: {context}
    Question: {question}
    Answer:""")
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": template}
    )
    return qa

# ── interface ─────────────────────────────────────────────────
st.title("Chatbot ISO 27001")
st.caption("Perguntas sobre a norma ISO 27001")

# histórico na sessão
if "historico" not in st.session_state:
    st.session_state.historico = []

# mostra histórico
for msg in st.session_state.historico:
    st.chat_message(msg["role"]).write(msg["content"])

# input
pergunta = st.chat_input("Sua pergunta:")

if pergunta:
    qa = carregar_pipeline()
    
    st.chat_message("user").write(pergunta)
    st.session_state.historico.append({"role": "user", "content": pergunta})
    
    with st.spinner("Buscando no manual..."):
        resposta = qa.invoke(pergunta)
    
    st.chat_message("assistant").write(resposta["result"])
    st.session_state.historico.append({
        "role": "assistant", 
        "content": resposta["result"]
    })
#%%

