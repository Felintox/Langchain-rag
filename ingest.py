#%%
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
#%%
# Carregar o PDF e criar os documentos
carregar_pdf= PyPDFLoader('data/iso27001.pdf')
documento=carregar_pdf.load()

#%%
# Criação do splitter ele vai dividir o documento em pedaços menores, para facilitar a busca e a recuperação de informações. O chunk_size define o tamanho máximo de cada pedaço, enquanto o chunk_overlap define quantos caracteres do final de um pedaço se repetem no início do próximo pedaço. Isso ajuda a garantir que as informações não sejam perdidas entre os pedaços. O length_function é usado para calcular o tamanho do chunk com base no comprimento do texto.
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000, #tamanho do chunk
    chunk_overlap=200, #quantos caracteres do chunk anterior se repetem no próximo
    length_function=len) # tamanho do chunk baseado no length do texto / poderia ser baseado no número de tokens

chunks=splitter.split_documents(documento)
# %%
print(f"Páginas originais: {len(documento)}")
print(f"Chunks gerados:    {len(chunks)}")
print("---\n")
print(f'{chunks[0].page_content} \n')   # texto do primeiro chunk
print(chunks[0].metadata)       # de qual página veio, nome do arquivo...
# %%
# Com o chunks criado queremos agora aplicar um embeddings nos chunks e armazenar os embeddings em um vetor store, para facilitar a busca e a recuperação de informações. O OllamaEmbeddings é uma classe que gera embeddings usando o modelo Ollama, e o Chroma é um vetor store que armazena os embeddings e permite realizar buscas eficientes.
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(chunks, embeddings, 
                           persist_directory="./db",
                           collection_metadata={"hnsw:space": "cosine"})
                           

# %%
results = db.similarity_search_with_relevance_scores(
    "What does ISO 27001 say about network segregation?", k=3
)

for doc, score in results:
    print(f"Score: {score:.3f}")
    print(f"Texto: {doc.page_content}")
    print("---")


# %%
template=PromptTemplate.from_template(
"Anwser to the question based on the context, if you don't know the answer say you don't know, don't try to invent an answer. \n\n"
"Context: {context}\n\n"
"Question: {question}\n\n"
)
# %%
llm = OllamaLLM(model="llama3.1:8b")
retriever = db.as_retriever(search_kwargs={"k": 3})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": template}  # aqui entra o seu template
)

resposta = qa.invoke("What does ISO 27001 say about access control?")
print(resposta["result"])
# %%
