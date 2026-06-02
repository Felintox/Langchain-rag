#%%
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
#%%
# Este Script tem como objetivo criar o vetor store a partir do PDF do artigo cientificado
# Além de realizar alguns teste de busca e recuperação de informações, para garantir que o vetor store foi criado corretamente e que as informações estão sendo recuperadas de forma relevante.
#%%
# Carregamento do PDF
carregar_pdf=PyMuPDFLoader('data/arc-mof.pdf')
documento=carregar_pdf.load()

#%%
for doc in documento:
    print("Pagina:", doc.metadata.get("page"))
    print("Page label:", doc.metadata.get("page_label"))
    print(doc.page_content)  # mostra os primeiros 500 caracteres do conteúdo da página
    print("---\n")
#%%
# Irémos fazer uma divisão entre main_core do artigo e dados de referencias
#%%
documentos_processados = []
achou_corte = False

for doc in documento:
    # Se já passou pela página do corte, tudo daqui pra frente é referência
    if achou_corte:
        doc.metadata["secao"] = "referencia"
        documentos_processados.append(doc)
        continue
        
    texto = doc.page_content
    pos = texto.upper().find("ASSOCIATED CONTENT")
    
    # Se encontrou a palavra, é hora de dividir a página atual em duas
    if pos != -1:
        # 1. Fatiando o texto original
        texto_principal = texto[:pos]
        texto_referencia = texto[pos:]
        
        # 2. Mantendo a página atual (ex: 13) só com a primeira metade
        doc.page_content = texto_principal
        doc.metadata["secao"] = "main_core"
        documentos_processados.append(doc)
        
        # 3. Criando uma página clone (ex: 13_1) para receber a segunda metade
        metadados_ref = doc.metadata.copy()
        metadados_ref["page"] = str(metadados_ref.get("page")) + "_1" # Vira "13_1"
        metadados_ref["secao"] = "referencia"
        
        nova_pagina_ref = Document(page_content=texto_referencia, metadata=metadados_ref)
        documentos_processados.append(nova_pagina_ref)
        
        achou_corte = True
    else:
        # Páginas normais antes do corte
        doc.metadata["secao"] = "main_core"
        documentos_processados.append(doc)

# No final, 'documentos_processados' terá a página 13 e a 13_1 separadas perfeitamente!

#%%
# Tamanho do documento:
for doc in documentos_processados:
    print(f"Página: {doc.metadata.get('page')}, Seção: {doc.metadata.get('secao')}, Tamanho: {len(doc.page_content)} caracteres")
#%%
# Definição do splitter e criação dos chunks
# Criação do splitter ele vai dividir o documento em pedaços menores, para facilitar a busca e a recuperação de informações. O chunk_size define o tamanho máximo de cada pedaço, enquanto o chunk_overlap define quantos caracteres do final de um pedaço se repetem no início do próximo pedaço. Isso ajuda a garantir que as informações não sejam perdidas entre os pedaços. O length_function é usado para calcular o tamanho do chunk com base no comprimento do texto.
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000, #tamanho do chunk
    chunk_overlap=200, #quantos caracteres do chunk anterior se repetem no próximo
    length_function=len) # tamanho do chunk baseado no length do texto / poderia ser baseado no número de tokens

chunks=splitter.split_documents(documentos_processados)
# %%
print(f"Páginas originais: {len(documentos_processados)}")
print(f"Chunks gerados:    {len(chunks)}")
print("---\n")
print(f'{chunks[0].page_content} \n')
print("---\n")   # texto do primeiro chunk
print(chunks[0].metadata)       # de qual página veio, nome do arquivo...
# %%
# Com o chunks criado queremos agora aplicar um embeddings nos chunks e armazenar os embeddings em um vetor store, para facilitar a busca e a recuperação de informações. O OllamaEmbeddings é uma classe que gera embeddings usando o modelo Ollama, e o Chroma é um vetor store que armazena os embeddings e permite realizar buscas eficientes.
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
db = Chroma.from_documents(chunks, embeddings, 
                           persist_directory="./db",
                           collection_metadata={"hnsw:space": "cosine"})
                           

# %%
# Testando o banco de dados vetorial
results = db.similarity_search_with_relevance_scores(
    "What is the ARC-MOF database?", k=5,filter = {'secao':'main_core'}
)

for doc, score in results:
    print(f"Score: {score:.3f}")
    print(f"Texto: {doc.page_content}")
    print("---")

# %%
#Teste da LLM local:
template=PromptTemplate.from_template(
"Anwser to the question based on the context, if you don't know the answer say you don't know, don't try to invent an answer. \n\n"
"Context: {context}\n\n"
"Question: {question}\n\n"
)
llm = OllamaLLM(model="llama3.1:8b")
retriever = db.as_retriever(search_kwargs={"k": 3})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": template}  # aqui entra o seu template
)
resposta = qa.invoke("How they get co2 adsorption capacity of a material?")
print(resposta["result"])
# %%
