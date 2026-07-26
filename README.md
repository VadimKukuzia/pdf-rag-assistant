# RAG Assistant

> **Production-ready RAG-система** на базі **LangGraph**, **FastAPI**, **ChromaDB** та **Gemini API** з інтегрованим трейсингом у **LangSmith** та веб-інтерфейсом на **Streamlit**.

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/LangGraph-Orchestration-orange)
![Backend](https://img.shields.io/badge/FastAPI-REST_API-green)
![Frontend](https://img.shields.io/badge/Streamlit-UI-red)

---

## 📌 Про проєкт

**RAG Assistant** — це інтелектуальний асистент для пошуку та аналізу інформації у локальних PDF-документах. Проєкт реалізовано з урахуванням сучасних паттернів побудови LLM-додатків: графова оркестрація стану, векторний пошук, обробка квот API та повний моніторинг виконання запитів.

### ✨ Ключові можливості:
* **Графова оркестрація (LangGraph):** Процес обробки запиту розбитий на ізольовані вузли з чітким управлінням станом (`GraphState`).
* **Векторний пошук (ChromaDB):** Семантичний пошук релевантних контекстних чанків з автоматичною дедуплікацією.
* **Захист від квот (Quota Guardrails):** Оптимізована кількість викликів LLM та обробка помилок `429 (Resource Exhausted)` без падіння сервера.
* **Асинхронний REST API (FastAPI):** Ендпоінти для генерації відповідей, завантаження документів та перевірки стану системи (`/health`).
* **Повна телеметрія (LangSmith):** Детальне наскрізне тресування виконання кожного вузла графа (затримка, токени, промпти).
* **Зручний UI (Streamlit):** Двопанельний веб-інтерфейс з інтерактивним чатом, показом джерел контексту та модулем індексації PDF.

---

## Архітектура системи

```text
[ Користувач / Streamlit UI ]
             │
             ▼
     [ FastAPI Backend ] ─── ( /api/v1/query, /api/v1/upload )
             │
             ▼
    [ LangGraph Pipeline ] ─── (Моніторинг: LangSmith)
             │
      ┌──────┴─────────────────────────────┐
      │  1. Guard Node (Безпека)           │
      │  2. Rephraser Node (Перефразування)│
      │  3. Retriever Node (ChromaDB)      │
      │  4. Generator Node (Gemini API)    │
      │  5. Validator Node (Валідація)     │
      └────────────────────────────────────┘
```

---

## 🛠️ Стек технологій

| Компонент | Технологія |
| --- | --- |
| **LLM & Embeddings** | Google Gemini API (`gemini-3.1-flash-lite`, `models/gemini-embedding-001`) |
| **Orchestration** | LangGraph, LangChain Core |
| **Vector Database** | ChromaDB |
| **Backend Framework** | FastAPI, Uvicorn, Pydantic |
| **Frontend UI** | Streamlit |
| **Observability** | LangSmith |

---

## 🚀 Швидкий запуск

### 1. Клонування репозиторію та створення venv

```bash
git clone 
cd rag-assistant

python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 2. Встановлення залежностей

```bash
pip install -r requirements.txt
```

### 3. Налаштування змінних середовища (`.env`)

Створіть файл `.env` у корені проєкту та вкажіть свої ключі:

```env
GEMINI_API_KEY="your_gemini_api_key_here"

# Налаштування LangSmith Telemetry
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="your_langsmith_api_key_here"
LANGCHAIN_PROJECT="rag-assistant"
```

### 4. Запуск FastAPI Backend

```bash
uvicorn main:app --reload
```

* Swagger UI буде доступний за адресою: `http://127.0.0.1:8000/docs`

### 5. Запуск Streamlit Frontend

У новому вікні термінала виконайте:

```bash
streamlit run app.py
```

* Інтерфейс відкриється за адресою: `http://localhost:8501`
---

## 📸 Скріншоти та демонстрація

### 💬 Інтерактивний чат із джерелами



### 📂 Панель завантаження та індексації PDF



### 🌳 Трейсинг у LangSmith



---

## 📝 Ліцензія

Проєкт розповсюджується під ліцензією MIT.
