#%%
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

app = FastAPI()

qa_chain= None

@app.lifespan("startup")
def iniciar_pipeline():
    global qa_chain

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma(persist_directory="./db", embedding_function=embeddings)
    
    retriever = db.as_retriever(search_kwargs={"k": 3, "filter": {"secao": "main_core"}})
    llm = OllamaLLM(model="llama3.1:8b")
    
    template = PromptTemplate.from_template(
        "You are an expert AI assistant specializing in materials science and chemistry.\n"
        "Use ONLY the following retrieved context to answer the user's question.\n"
        "If the answer is not contained in the context, explicitly say 'I don't know based on the provided text'. "
        "Do not make up information or use prior knowledge.\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Helpful Answer:"
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type_kwargs={"prompt": template}
    )

class Pergunta(BaseModel):
    pergunta: str

@app.post('/pergunta')
def fazer_pergunta(requisicao: Pergunta):
    resposta = qa_chain.invoke(requisicao.pergunta)
    return {"resposta": resposta["result"]}
