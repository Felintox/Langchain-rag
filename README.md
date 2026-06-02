# RAG para Artigos Científicos - Arquitetura Cliente-Servidor com FastAPI e Streamlit

Este projeto implementa um sistema de Retrieval-Augmented Generation (RAG) voltado para a extração de conhecimento de artigos científicos complexos (focado em Ciência dos Materiais e Química, utilizando o banco de dados ARC-MOF como exemplo).

A aplicação adota uma **arquitetura de produção (Cliente-Servidor)**, separando o motor de Inteligência Artificial (Backend/API) da interface do usuário (Frontend), além de aplicar técnicas avançadas de pré-processamento de dados para mitigar alucinações.

## Visão Geral e Diferenciais

Diferente de implementações RAG ingênuas (*Naive RAG*), este projeto incorpora práticas avançadas exigidas pela indústria:

1. **Pré-processamento e Limpeza (Data Cleansing):** Uso do `PyMuPDFLoader` para extração de alta fidelidade e fatiamento automático de páginas para separar conteúdo principal de bibliografias/referências.
2. **Filtragem por Metadados:** Os chunks são tagueados (`secao: main_core` ou `secao: referencia`), permitindo que a busca ignore conteúdos irrelevantes e aumente a precisão da resposta.
3. **Embeddings Otimizados para QA:** Substituição de embeddings genéricos pelo modelo `BAAI/bge-small-en-v1.5`, especializado em mapeamento de perguntas e respostas.
4. **API RESTful (Backend):** Servidor FastAPI que mantém a LLM (`llama3.1:8b`) e a base vetorial carregados na memória de forma global, garantindo alta velocidade de resposta sem gargalos de leitura/escrita em disco.
5. **Frontend Leve:** Interface Streamlit operando como um cliente puro via requisições HTTP, simulando um ambiente real onde a interface gráfica roda separada da infraestrutura de IA.

## Arquitetura do Sistema

```text
[ Pipeline de Ingestão de Dados ]
 Artigo Científico PDF -> PyMuPDFLoader -> Slicing/Metadados -> TextSplitter -> HuggingFace BGE Embeddings -> ChromaDB

[ Backend / Servidor de IA ]
 FastAPI (main.py) <--- Consulta (Retriever + Filtro 'main_core') ---> ChromaDB
       |
       +---> Prompt Template Otimizado (Especialista em Materiais) ---> Ollama (Llama 3.1:8B)
       |
       v
  (JSON via HTTP POST)
       |
[ Frontend / Interface ]
 Streamlit (app.py) ---> Usuário Final
```

## Estrutura do Projeto

```text
.
├── main.py        # Backend: API FastAPI com a lógica da LLM e RAG
├── app.py         # Frontend: Interface leve com Streamlit
├── ingest.py      # Pipeline: Limpeza de dados, metadados e criação do Vector Store
├── data/
│   └── arc-mof.pdf
└── db/            # Base vetorial persistida localmente (Chroma)
```

## Tecnologias Utilizadas

- **Linguagem:** Python
- **Backend da API:** FastAPI, Uvicorn, Pydantic
- **Frontend:** Streamlit, Requests
- **Orquestração de LLM:** LangChain
- **Banco de Dados Vetorial:** ChromaDB
- **Modelos Locais:** Ollama (`llama3.1:8b`)
- **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`)
- **Processamento de PDF:** PyMuPDF (`fitz`), RecursiveCharacterTextSplitter

## Como Executar

### 1. Instalar dependências

```bash
pip install fastapi uvicorn pydantic streamlit requests langchain langchain-community langchain-classic langchain-text-splitters langchain-ollama chromadb pymupdf sentence-transformers
```

### 2. Baixar o modelo de inferência no Ollama

Certifique-se de que o Ollama está instalado na sua máquina e inicie o download do modelo:
```bash
ollama pull llama3.1:8b
```

### 3. Ingerir Dados e Gerar a Base Vetorial

```bash
python ingest.py
```
*Este comando processará o PDF, separará o conteúdo principal das referências com base na palavra-chave "ASSOCIATED CONTENT", gerará os embeddings e salvará os dados no diretório local `./db`.*

### 4. Ligar a API (Backend)

Em um terminal, inicie o servidor:
```bash
uvicorn main:app --reload
```
*Aguarde a inicialização do pipeline (quando a LLM e a base Chroma são carregadas em memória).*

### 5. Ligar a Interface do Usuário (Frontend)

Com o servidor rodando, abra um **novo terminal** e inicie o Streamlit:
```bash
streamlit run app.py
```

## Exemplos de Perguntas

Para testar o conhecimento extraído, tente enviar:
- *What is the ARC-MOF database?*
- *How they get CO2 adsorption capacity of a material?*
- *Which machine learning models are used in this context?*

## Roadmap e Evoluções Planejadas

Embora o sistema atual já contemple arquitetura avançada de metadados e separação de cliente-servidor, os próximos passos visam escalar o projeto para Nível Enterprise (Produção Plena):

- **Segurança e Autenticação:** Implementar validação via API Key / OAuth2 nas rotas do FastAPI.
- **Reranking / Cross-Encoders:** Adicionar uma camada de reordenação (ex: Cohere Rerank) logo após a busca do ChromaDB para refinar ainda mais os chunks recuperados.
- **Observabilidade:** Integrar ferramentas de tracing como LangSmith ou Arize Phoenix para monitorar chamadas LLM e latência do Retrieval.
- **Memória Conversacional:** Adicionar endpoints na API e na interface capazes de passar o `chat_history` no escopo da chain para permitir follow-up questions.
- **Avaliação Automatizada:** Usar *LLM-as-a-Judge* (ex: framework Ragas) para quantificar as métricas de *Context Precision* e *Faithfulness* do RAG.
