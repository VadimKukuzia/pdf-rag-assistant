# RAG Assistant

> **Production-Ready Conversational RAG-Агент** на базі **LangGraph**, **FastAPI**, **Hybrid Search (ChromaDB + BM25)** та **Gemini API** з інтелектуальним викликом інструментів (Tool-Calling), збереженням історії сесій в **SQLite**, повною контейнеризацією в **Docker** та трейсингом у **LangSmith**.

![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue)
![Framework](https://img.shields.io/badge/LangGraph-Orchestration-orange)
![Backend](https://img.shields.io/badge/FastAPI-REST_API-green)
![Frontend](https://img.shields.io/badge/Streamlit-UI-red)

## 📌 Про проєкт

**RAG Assistant** — це модульна агентна система для ведення діалогу та інтелектуального пошуку інформації у PDF-документах.

На відміну від класичних жорстких RAG-пайплайнів, у цій версії реалізовано паттерн **Conversational Tool-Calling Agent**:

1. **Звичайні питання та привітання** агент обробляє самостійно в межах контексту бесіди, заощаджуючи токени та час.
2. **Питання щодо змісту PDF** автоматично тригерують виклик інструменту `search_pdf_documents`, який запускає ізольований графовий **Hybrid RAG Pipeline** (Sparse + Dense Search).


## 🏗️ Архітектура системи

```text
[ Користувач / Streamlit UI ]
              │
              ▼
     [ FastAPI Backend ] ─── (/api/v1/chat, /api/v1/upload, /health)
              │
              ▼
 [ Conversational Agent ] ─── (Збереження історії: SQLite)
              │
     (Потрібен контекст з PDF?)
              ├── Ні ───► Пряма відповідь користувачу
              │
              └── Так ──► [ Tool: search_pdf_documents ]
                                      │
                                      ▼
                           [ LangGraph RAG Pipeline ]
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           │ 1. Guard Node (Аналіз безпеки та Prompt Injection)  │
           │ 2. Rephraser Node (Оптимізація та перефразування)   │
           │ 3. Hybrid Retriever (Ensemble: ChromaDB + BM25)     │
           │ 4. Generator Node (Синтез відповіді Gemini API)     │
           │ 5. Validator Node (Перевірка галюцинацій)           │
           └─────────────────────────────────────────────────────┘

```

---

## 📁 Структура проєкту

```text
rag-assistant/
├── app/
│   ├── agent/          # AI-Агент та обгортки інструментів (Tools)
│   │   ├── agent.py
│   │   └── tools.py
│   ├── api/            # FastAPI REST API ендпоінти
│   │   └── main.py
│   ├── core/           # Конфігурації (Pydantic Settings, Async DB, policy.yaml)
│   │   ├── config.py
│   │   └── database.py
│   ├── model/          # Async SQLAlchemy моделі та Pydantic DTO / GraphState
│   │   ├── models.py
│   │   └── schemas.py
│   └── rag/            # Ядро RAG (LangGraph nodes, graph, hybrid retriever, ingestion)
│       ├── graph.py
│       ├── ingestion.py
│       ├── nodes.py
│       └── retriever.py
├── data/               # Приклади PDF-документів
├── tests/              # Набір автотестів (Pytest)
│   ├── test_api.py
│   ├── test_graph.py
│   ├── test_ingestion.py
│   └── test_nodes.py
├── ui/                 # Streamlit чат-інтерфейс
│   └── chat.py
├── .dockerignore
├── .env.example - мінімально необхідний для запуску файл середовища
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── README.md

```

---

## 🛠️ Стек технологій

| Компонент | Технологія |
| --- | --- |
| **LLM & Embeddings** | Google Gemini API (`gemini-3.1-flash-lite`, `models/gemini-embedding-001`) |
| **Agent & Orchestration** | LangChain Agent (Tool-Calling), LangGraph, LangChain Core |
| **Retriever Engine** | Hybrid Search (`EnsembleRetriever`: Dense ChromaDB + Sparse BM25) |
| **Storage & DB** | ChromaDB (Vector DB), SQLite + AsyncSQLAlchemy / aiosqlite (Chat History) |
| **Backend Framework** | FastAPI, Uvicorn, Pydantic v2 |
| **Frontend UI** | Streamlit |
| **Containerization** | Docker, Docker Compose |
| **Testing** | Pytest, TestClient |
| **Observability** | LangSmith (Full Execution Tracing) |

---

## 🚀 Швидкий запуск

### Варіант 1. Запуск через Docker Compose (Рекомендований)

Для запуску всієї інфраструктури (FastAPI + Streamlit + ChromaDB + SQLite) потрібен лише встановлений **Docker Desktop**.

1. **Клонуйте репозиторій:**
```bash
git clone https://github.com/VadimKukuzia/pdf-rag-assistant.git
cd pdf-rag-assistant

```


2. **Створіть `.env` файл у корені проєкту:**
```env
GEMINI_API_KEY="your_gemini_api_key_here"

# Налаштування LangSmith Телеметрії (опціонально)
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your_langsmith_api_key_here"
LANGCHAIN_PROJECT="rag-assistant"

```


3. **Запустіть контейнери:**
```bash
docker compose up --build

```



* **Streamlit Web UI:** `http://localhost:8501`
* **FastAPI Interactive Docs (Swagger):** `http://localhost:8000/docs`
* **API Health Check:** `http://localhost:8000/health`

---

### Варіант 2. Локальний запуск для розробки

1. **Створіть та активуйте віртуальне середовище:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

```


2. **Встановіть залежності:**
```bash
pip install -r requirements.txt

```


3. **Запустіть FastAPI Backend:**
```bash
python -m uvicorn app.api.main:app --reload

```


4. **Запустіть Streamlit UI (у новому терміналі):**
```bash
streamlit run ui/chat.py

```


5. **Запуск автотестів:**
```bash
python -m pytest -v

```



---

## 📸 Скріншоти та демонстрація роботи

### 💬 Спілкування з Агентом (покроковий сценарій)

1. **Старт бесіди та звичайний діалог (БЕЗ виклику RAG-інструменту):**

2. **Виклик при відсутньому файлі:**

3. **Завантаження та обробка файлу:**

4. **Відповідь на основі контексту з джерелами:**

5. **Відхилення спроби Prompt Injection / Небезпечного запиту:**


---

### 🌳 Трейсинг виконання в LangSmith

На скріншоті видно наскрізний виклик: від звернення до Агента, виклику `search_pdf_documents` до виконання всіх 5 нод графа LangGraph (Guard -> Rephraser -> EnsembleRetriever [Dense + BM25] -> Generator -> Validator):

---

## 📝 Ліцензія

Проєкт розповсюджується під ліцензією MIT.