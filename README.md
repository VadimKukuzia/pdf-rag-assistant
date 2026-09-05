# RAG Assistant

> **Conversational RAG Agent with Tool Calling & Hybrid Search** powered by **LangGraph**, **FastAPI**, **ChromaDB + BM25**, and the **Google Gemini API**.  
> Features autonomous tool calling, persistent session history via **Async SQLAlchemy + SQLite**, containerization with **Docker**, and full observability via **LangSmith**.

[![CI Pipeline](https://github.com/VadimKukuzia/pdf-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/VadimKukuzia/pdf-rag-assistant/actions/workflows/ci.yml)
![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue)
![Framework](https://img.shields.io/badge/LangGraph-Orchestration-orange)
![Backend](https://img.shields.io/badge/FastAPI-REST_API-green)
![Frontend](https://img.shields.io/badge/Streamlit-UI-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Overview

**RAG Assistant** is a modular agentic system designed for multi-turn dialogue and precise context-grounded document question answering across uploaded PDF files.

Unlike rigid linear RAG pipelines, this implementation leverages the **Conversational Tool-Calling Agent** pattern:
1. **Chit-chat & General Inquiries:** The agent answers directly without activating knowledge retrieval tools.
2. **Context-Dependent Queries:** The agent autonomously triggers the `search_pdf_documents` tool, initiating a dedicated **Hybrid RAG Pipeline** (Sparse BM25 + Dense Semantic Search).

## 🏗️ System Architecture

```text
[ User / Streamlit Web UI ]
             │
             ▼
    [ FastAPI Backend ] ─── (/api/v1/chat, /api/v1/upload, /health)
             │
             ▼
  [ Conversational Agent ] ─── (Persistent History: SQLite / AsyncSQLAlchemy)
             │
     (Requires PDF Context?)
             ├── No ────► Direct LLM Response
             │
             └── Yes ───► [ Tool: search_pdf_documents ]
                                      │
                                      ▼
                          [ LangGraph RAG Pipeline ]
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           │ 1. Guard Node (Security check & Prompt Injection)   │
           │ 2. Rephraser Node (Query decontextualization)       │
           │ 3. Hybrid Retriever (Ensemble: ChromaDB + BM25)     │
           │ 4. Generator Node (Contextual Synthesis via Gemini) │
           │ 5. Validator Node (Hallucination Detection Guard)   │
           └─────────────────────────────────────────────────────┘

```

---

## 📁 Project Structure

```text
rag-assistant/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline (lint & tests)
├── app/
│   ├── agent/                  # Conversational Agent & tool definitions
│   │   ├── agent.py
│   │   └── tools.py
│   ├── api/                    # FastAPI REST routes & application lifecycle
│   │   └── main.py
│   ├── core/                   # App configurations, async DB engine & policy loader
│   │   ├── config.py
│   │   └── database.py
│   ├── model/                  # SQLAlchemy ORM models, Pydantic schemas, GraphState
│   │   ├── models.py
│   │   └── schemas.py
│   └── rag/                    # Core RAG components (LangGraph, nodes, hybrid retriever)
│       ├── bm25_cache.py
│       ├── graph.py
│       ├── ingestion.py
│       ├── nodes.py
│       └── retriever.py
├── data_sample/                # Sample PDF documents for verification
├── tests/                      # Automated test suite
│   ├── check_ingestion.py
│   ├── check_setup.py
│   ├── test_api.py
│   ├── test_graph.py
│   └── test_nodes.py
├── ui/                         # Streamlit chat user interface
│   └── chat.py
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── policy.yaml                 # Security rules, chunking parameters & prompts
├── pytest.ini                  # Pytest configuration & paths
├── requirements.txt
├── ruff.toml                   # Ruff linter rules & ignores
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
| --- | --- |
| **LLM & Embeddings** | Google Gemini API (`gemini-3.1-flash-lite`, `models/gemini-embedding-001`) |
| **Agent & Orchestration** | LangChain Agent (Tool-Calling), LangGraph, LangChain Core |
| **Retriever Engine** | Hybrid Search (`EnsembleRetriever`: Dense ChromaDB + Sparse BM25) |
| **Storage & DB** | ChromaDB (Vector DB), SQLite + AsyncSQLAlchemy / aiosqlite (Chat History) |
| **Backend Framework** | FastAPI, Uvicorn, Pydantic v2 |
| **Frontend UI** | Streamlit |
| **Containerization** | Docker, Docker Compose |
| **Testing & CI/CD** | Pytest, TestClient, Ruff, GitHub Actions |
| **Observability** | LangSmith (Full Execution Tracing) |

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)
Requires only **Docker Desktop** installed.

1. **Clone the repository:**
```bash
git clone https://github.com/VadimKukuzia/pdf-rag-assistant.git
cd pdf-rag-assistant

```


2. **Configure environment variables in .env:**
```env
GEMINI_API_KEY="your_gemini_api_key_here"

# Optional: LangSmith Tracing
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your_langsmith_api_key_here"
LANGCHAIN_PROJECT="your_langchain_project_name"

```


3. **Run the services:**
```bash
docker compose up --build

```



* **Streamlit Web UI:** `http://localhost:8501`
* **FastAPI Interactive Docs (Swagger):** `http://localhost:8000/docs`
* **API Health Check:** `http://localhost:8000/health`

---

### Option 2: Local Setup

1. **Create and activate a virtual environment:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Start the FastAPI backend:**
```bash
python -m uvicorn app.api.main:app --reload

```


4. **Start the Streamlit interface (in a new terminal):**
```bash
streamlit run ui/chat.py

```


5. **Run test suite:**
```bash
python -m pytest -v

```



---

## 📸 Demonstration & Screenshots
> *Note: Demonstrated with a sample Ukrainian document.*
### 💬 Agent Workflow

1. **Direct chit-chat without retrieving documents:**
<img width="1763" height="955" alt="Screenshot_13-8-2026_0131_localhost" src="https://github.com/user-attachments/assets/6e50cc8b-7666-4daa-a8dd-a799328bb15d" />

2. **Question try without documents:**
<img width="1763" height="955" alt="Screenshot_13-8-2026_01347_localhost" src="https://github.com/user-attachments/assets/e5e4ce7a-2478-4811-9350-4654166bcf5f" />

3. **PDF ingestion & parsing:**
<img width="1763" height="955" alt="Screenshot_13-8-2026_01614_localhost" src="https://github.com/user-attachments/assets/04b43f80-89c4-4d77-9225-753e84e2922d" />

4. **Context-grounded answer generation with source attribution:**
<img width="1763" height="955" alt="Screenshot_13-8-2026_01647_localhost" src="https://github.com/user-attachments/assets/114d334c-f3bc-4446-b0f2-4f59b711fadd" />

5. **Security Guardrail (Prompt Injection Mitigation):**
<img width="1763" height="955" alt="Screenshot_13-8-2026_0212_localhost" src="https://github.com/user-attachments/assets/d6ee2ae9-e3ca-4247-84da-6077c110bc15" />


---

### 🌳 Observability with LangSmith

Full execution graph trace: tracking the journey from Agent trigger to tool invocation through all 5 LangGraph workflow stages (Guard → Rephraser → EnsembleRetriever → Generator → Validator):
<img width="1351" height="781" alt="Screenshot_13-8-2026smithjpeg" src="https://github.com/user-attachments/assets/dbb0b2c3-a64a-4f1c-8243-c74c210e8425" />


---

## 📝 License

Distributed under the MIT License. See LICENSE for more details.
