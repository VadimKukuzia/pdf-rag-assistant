from dotenv import load_dotenv
load_dotenv()

import os
import re
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from config import policy
from schemas import QueryRequest, QueryResponse, IngestResponse, SourceChunk
from ingestion import ingest_pdf
from graph import run_rag_pipeline

# Ініціалізація FastAPI додатку з метаданими для Swagger UI
app = FastAPI(
    title="RAG Assistant API 🚀",
    description=(
        "Production-ready REST API для інтелектуального пошуку в документації "
        "на основі LangGraph, Google Gemini API, ChromaDB та Guardrails."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Налаштування CORS (щоб до API можна було звертатися з веб-фронтенду або Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 1. СИСТЕМНІ ЕНДПОІНТИ
# ==========================================
@app.get("/health", tags=["System"], summary="Перевірка статусу API")
async def health_check():
    """Повертає статус готовності сервісу."""
    return {"status": "ok", "service": "RAG Assistant API"}


# ==========================================
# 2. ОСНОВНІ БІЗНЕС-ЕНДПОІНТИ
# ==========================================
@app.post(
    "/api/v1/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Завантажити та проіндексувати PDF-документ",
    tags=["Ingestion"]
)
async def upload_file(file: UploadFile = File(...)):
    """
    Приймає PDF-файл, перевіряє розширення та розмір, 
    нарізає на чанки та зберігає ембедінги у ChromaDB.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Непідтримуваний формат файлу. Дозволено завантажувати тільки .pdf"
        )

    temp_file_path = f"temp_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = ingest_pdf(temp_file_path)
        result["filename"] = file.filename
        
        return IngestResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Помилка під час індексації документа: {str(e)}"
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Надіслати запитання до RAG-пайплайну",
    tags=["RAG Engine"]
)
async def query_rag(request: QueryRequest):
    """
    Приймає запитання користувача, запускає LangGraph-пайплайн 
    (Guard -> Rephraser -> Retriever -> Generator -> Validator) та повертає відповідь із джерелами.
    """
    try:
        final_state = run_rag_pipeline(
            query=request.query,
            session_id=request.session_id or "default_session"
        )

        sources = []
        for doc in final_state.get("documents", []):
            raw_source = doc.metadata.get("source_file") or doc.metadata.get("source", "документ.pdf")
            clean_filename = os.path.basename(raw_source).replace("temp_", "")

            raw_preview = doc.page_content[:350]
            clean_preview = re.sub(r'\s+', ' ', raw_preview).strip() + "..."

            sources.append(
                SourceChunk(
                    source_file=clean_filename,
                    content_preview=clean_preview
                )
            )

        is_safe = final_state.get("is_safe", True)
        answer = final_state.get("generation") or policy["responses"]["fallback_no_context"]

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            is_safe=is_safe,
            status="success" if is_safe else "rejected"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Помилка під час обробки запиту у RAG-пайплайні: {str(e)}"
        )