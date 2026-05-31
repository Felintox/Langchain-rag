#%%
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
#%%
# Carregar o PDF e criar os documentos
carregar_pdf= PyPDFLoader('data/manual_siemens_fanuc.pdf')
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
db = Chroma.from_documents(chunks, embeddings, persist_directory="./db")

#%%
# Testando a busca no vetor store. A função db.similarity_search recebe uma consulta e retorna os chunks mais relevantes com base na similaridade dos embeddings.
results = db.similarity_search_with_relevance_scores(
    "Qual o torque máximo do eixo X?", k=3
)
#%%
for doc, score in results:
    print(f"Score: {score:.3f}")
    print(f"Página: {doc.metadata['page']}")
    print(f"Texto: {doc.page_content[:200]}")
    print("---")
# %%
print(db._collection.count())  # quantos chunks tem
print(db._embedding_function)  # qual modelo de embedding está sendo usado
# %%
results = db.similarity_search_with_score(
    "programação paramétrica CNC", k=3
)

for doc, score in results:
    print(f"Score: {score:.3f}")
    print(f"Texto: {doc.page_content[:150]}")
    print("---")
# %%
for i, doc in enumerate(documento[:10]):
    print(f"Página {i}: {len(doc.page_content)} caracteres")
    print(doc.page_content[:100])
    print("---")
# %%
