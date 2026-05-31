# LangChain RAG - Chatbot ISO 27001

Este projeto e um chatbot RAG para consulta de documentos PDF usando LangChain, Ollama, Chroma e Streamlit.

A aplicacao permite fazer perguntas sobre um documento de referencia, neste caso um PDF sobre ISO 27001, e gerar respostas com base nos trechos mais relevantes recuperados da base vetorial.

## Visao geral

O projeto implementa um fluxo RAG completo:

1. Carrega um PDF.
2. Divide o conteudo em chunks.
3. Gera embeddings dos trechos.
4. Salva os vetores em uma base Chroma local.
5. Recupera os chunks mais relevantes para cada pergunta.
6. Envia o contexto recuperado para uma LLM local via Ollama.
7. Exibe a resposta em uma interface de chat com Streamlit.

Esse fluxo permite que a LLM responda usando uma fonte externa de conhecimento, em vez de depender apenas do conhecimento interno do modelo.

## Arquitetura

```text
PDF ISO 27001
  |
  v
PyPDFLoader
  |
  v
RecursiveCharacterTextSplitter
  |
  v
OllamaEmbeddings
  |
  v
Chroma Vector Store
  |
  v
Retriever
  |
  v
Prompt com contexto
  |
  v
LLM Ollama
  |
  v
Resposta no Streamlit
```

## Estrutura do projeto

```text
.
+-- app.py
+-- rag.py
+-- data/
|   +-- iso27001.pdf
+-- db/
    +-- base vetorial Chroma
```

## Arquivos principais

### `rag.py`

Responsavel pela criacao da base de conhecimento.

Ele carrega o PDF, divide o texto em chunks, gera embeddings com `nomic-embed-text` e salva os vetores no Chroma dentro da pasta `db`.

Tambem possui testes simples de recuperacao para validar se a busca semantica esta retornando trechos relevantes.

### `app.py`

Responsavel pela interface do chatbot.

Ele carrega a base vetorial ja criada, configura o retriever, conecta a LLM `llama3.1:8b` e disponibiliza uma interface em Streamlit para o usuario fazer perguntas sobre o documento.

O historico da conversa aparece visualmente na tela usando `st.session_state`.

## Tecnologias utilizadas

- Python
- Streamlit
- LangChain
- Chroma
- Ollama
- `llama3.1:8b`
- `nomic-embed-text`
- PyPDFLoader
- RecursiveCharacterTextSplitter

## Como executar

### 1. Instalar dependencias

```bash
pip install streamlit langchain langchain-community langchain-classic langchain-text-splitters langchain-ollama chromadb pypdf
```

### 2. Baixar os modelos no Ollama

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 3. Gerar a base vetorial

```bash
python rag.py
```

Esse comando processa o PDF, cria os chunks, gera embeddings e salva a base no diretorio `db`.

### 4. Rodar o chatbot

```bash
streamlit run app.py
```

## Exemplos de perguntas

- What does ISO 27001 say about access control?
- What does ISO 27001 say about network segregation?
- How does ISO 27001 approach risk assessment?
- What are the requirements related to information security policies?

## Melhorias planejadas

O projeto ja possui o fluxo principal de RAG funcionando. As proximas melhorias planejadas sao:

- Migrar o vector store local para Qdrant.
- Adicionar historico conversacional real para a LLM.
- Reescrever perguntas de acompanhamento antes da busca.
- Exibir fontes, paginas e trechos usados na resposta.
- Melhorar o prompt para respostas em portugues e com maior controle de formato.
- Melhorar a estrategia de chunking com mais metadados.
- Adicionar suporte a multiplos documentos.
- Criar um pipeline separado de ingestao, como `ingest.py`.
- Adicionar logs dos chunks recuperados e dos scores de similaridade.
- Criar testes e perguntas de avaliacao para medir a qualidade das respostas.

## Sobre o tipo de RAG

Este projeto pode ser classificado como um RAG inicial, tambem chamado de Naive RAG no sentido tecnico.

Isso significa que ele implementa o fluxo essencial de RAG de forma direta: ingestao, chunking, embeddings, armazenamento vetorial, recuperacao semantica e geracao com contexto.

Essa e uma base solida para evoluir para uma arquitetura mais completa, com Qdrant, memoria conversacional, reranking, avaliacao, fontes e observabilidade.

## Status

Status atual: RAG local funcional para consulta de PDF sobre ISO 27001.

Proxima etapa recomendada: adicionar fontes nas respostas e migrar a base vetorial para Qdrant.
