import logging
import os
import re
import shutil
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.agent import run_agent_chat
from app.core.config import policy
from app.core.database import get_db, init_db
from app.model.models import ChatMessageModel, ChatSessionModel
from app.model.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    ChatMessage,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SessionHistoryResponse,
    SourceChunk,
)
from app.rag.graph import run_rag_pipeline
from app.rag.ingestion import ingest_pdf

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="RAG Assistant API 🚀",
    description="REST API для пошуку в документації з Agent Tool-Calling та збереженням сесій.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok", "service": "RAG Assistant API"}


@app.post(
    "/api/v1/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
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
        logger.error("Error during PDF ingestion: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Помилка під час індексації документа: {e!s}"
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post(
    "/api/v1/chat",
    response_model=AgentChatResponse,
    status_code=status.HTTP_200_OK,
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
    answer = str(agent_output.get("answer", ""))
    used_tools = agent_output.get("used_tools", [])

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
    tags=["Legacy RAG Engine"]
)
async def query_rag(request: QueryRequest):
    try:
        final_state = run_rag_pipeline(
            query=request.query,
            session_id="default_session"
        )

        sources = []
        docs = final_state.get("retrieved_docs") or final_state.get("documents", [])
        for doc in docs:
            raw_source = doc.metadata.get("source_file") or doc.metadata.get("source", "документ.pdf")
            clean_filename = os.path.basename(raw_source).replace("temp_", "")
            clean_preview = re.sub(r'\s+', ' ', doc.page_content[:350]).strip() + "..."

            sources.append(
                SourceChunk(source_file=clean_filename, content_preview=clean_preview)
            )

        answer = final_state.get("answer") or final_state.get("generation") or policy["responses"]["fallback_no_context"]

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            is_safe=final_state.get("is_safe", True),
            status="success"
        )
    except Exception as e:
        logger.error("Direct RAG query error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Помилка під час обробки запиту: {e!s}"
        )