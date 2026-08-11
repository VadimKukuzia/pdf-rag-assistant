from dotenv import load_dotenv
load_dotenv()

import os
import re
import shutil
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config import policy
from database import init_db, get_db
from models import ChatSessionModel, ChatMessageModel
from schemas import (
    QueryRequest, QueryResponse, IngestResponse, SourceChunk,
    AgentChatRequest, AgentChatResponse, ChatMessage, SessionHistoryResponse
)
from ingestion import ingest_pdf
from graph import run_rag_pipeline
from agent import run_agent_chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Життєвий цикл додатку: ініціалізація БД при запуску."""
    await init_db()
    yield


app = FastAPI(
    title="RAG Assistant API 🚀",
    description="REST API для пошуку в документації з Agent Tool-Calling та збереженням сесій.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"], summary="Перевірка статусу API")
async def health_check():
    return {"status": "ok", "service": "RAG Assistant API"}


@app.post(
    "/api/v1/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Завантажити та проіндексувати PDF-документ",
    tags=["Ingestion"]
)
async def upload_file(file: UploadFile = File(...)):
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
    "/api/v1/chat",
    response_model=AgentChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Спілкування з Агентом (збереження контексту сесії)",
    tags=["Conversational Agent"]
)
async def agent_chat(request: AgentChatRequest, db: AsyncSession = Depends(get_db)):
    session_id = request.session_id or "default_session"

    stmt = select(ChatSessionModel).where(ChatSessionModel.session_id == session_id)
    result = await db.execute(stmt)
    chat_session = result.scalars().first()

    if not chat_session:
        chat_session = ChatSessionModel(session_id=session_id)
        db.add(chat_session)
        await db.commit()

    msg_stmt = (
        select(ChatMessageModel)
        .where(ChatMessageModel.session_id == session_id)
        .order_by(ChatMessageModel.created_at.asc())
    )
    msg_result = await db.execute(msg_stmt)
    db_messages = msg_result.scalars().all()

    history = [{"role": m.role, "content": m.content} for m in db_messages]

    agent_output = run_agent_chat(message=request.message, history=history)
    answer = agent_output.get("answer", "")
    used_tools = agent_output.get("used_tools", [])

    # Перестраховка: гарантуємо, що answer — це string
    if not isinstance(answer, str):
        answer = str(answer)

    user_msg = ChatMessageModel(session_id=session_id, role="user", content=request.message)
    assistant_msg = ChatMessageModel(session_id=session_id, role="assistant", content=answer)
    db.add_all([user_msg, assistant_msg])
    await db.commit()

    return AgentChatResponse(
        session_id=session_id,
        answer=answer,
        used_tools=used_tools
    )

@app.get(
    "/api/v1/sessions/{session_id}/history",
    response_model=SessionHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Отримати історію повідомлень сесії",
    tags=["Conversational Agent"]
)
async def get_session_history(session_id: str, db: AsyncSession = Depends(get_db)):
    msg_stmt = (
        select(ChatMessageModel)
        .where(ChatMessageModel.session_id == session_id)
        .order_by(ChatMessageModel.created_at.asc())
    )
    msg_result = await db.execute(msg_stmt)
    messages = msg_result.scalars().all()

    return SessionHistoryResponse(
        session_id=session_id,
        messages=[ChatMessage(role=m.role, content=m.content) for m in messages]
    )


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Надіслати прямої запит до RAG-пайплайну",
    tags=["Legacy RAG Engine"]
)
async def query_rag(request: QueryRequest):
    try:
        final_state = run_rag_pipeline(
            question=request.query,
            collection_name="docs"
        )

        sources = []
        for doc in final_state.get("retrieved_docs", []):
            raw_source = doc.metadata.get("source_file") or doc.metadata.get("source", "документ.pdf")
            clean_filename = os.path.basename(raw_source).replace("temp_", "")
            raw_preview = doc.page_content[:350]
            clean_preview = re.sub(r'\s+', ' ', raw_preview).strip() + "..."

            sources.append(
                SourceChunk(source_file=clean_filename, content_preview=clean_preview)
            )

        answer = final_state.get("answer") or policy["responses"]["fallback_no_context"]

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            is_safe=True,
            status="success"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Помилка під час обробки запиту: {str(e)}"
        )