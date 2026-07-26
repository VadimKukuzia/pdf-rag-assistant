from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class QueryRequest(BaseModel):
    """Схема вхідного запиту користувача до RAG API."""
    
    query: str = Field(
        ...,
        description="Текст запитання користувача до RAG-асистента",
        examples=["Які умови відкриття картки?"],
        min_length=3,
        max_length=1000
    )
    session_id: Optional[str] = Field(
        default="default_session",
        description="Ідентифікатор сесії або чату для трейсингу в LangSmith"
    )


class SourceChunk(BaseModel):
    """Інформація про знайдений фрагмент документа-джерела."""
    
    source_file: str = Field(..., description="Назва PDF-файлу, з якого взято чанк")
    content_preview: str = Field(..., description="Короткий витяг із тексту чанку")


class QueryResponse(BaseModel):
    """Схема вихідної відповіді API користувачеві."""
    
    query: str = Field(..., description="Початковий запит користувача")
    answer: str = Field(..., description="Фінальна згенерована відповідь від Gemini")
    sources: List[SourceChunk] = Field(default_factory=list, description="Список використаних джерел з ChromaDB")
    is_safe: bool = Field(default=True, description="Статус проходження перевірки Guardrail")
    status: str = Field(default="success", description="Статус виконання (success / rejected / error)")


class IngestResponse(BaseModel):
    """Схема відповіді після індексації PDF-файлу."""
    
    status: str = Field(..., description="Результат виконання індексації")
    filename: str = Field(..., description="Назва завантаженого файлу")
    chunks_created: int = Field(..., description="Кількість згенерованих та збережених чанків")
    collection_name: str = Field(..., description="Назва колекції в ChromaDB")


class GraphState(TypedDict):
    """
    Контейнер стану (State), який передається між вузлами (Nodes) графа LangGraph.
    
    Кожна нода отримує поточний GraphState, модифікує його та повертає далі за ланцюжком.
    """
    
    query: str                             # Початковий запит користувача
    documents: List[Document]              # Отримані з ChromaDB чанки
    generation: Optional[str]              # Підсумкова текстова відповідь від LLM
    is_safe: bool                          # Чи пройшов запит перевірку на prompt injection
    rejection_reason: Optional[str]        # Причина відхилення, якщо is_safe=False
    source_files: List[str]                # Перелік унікальних назв файлів-джерел