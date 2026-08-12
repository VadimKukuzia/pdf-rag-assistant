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


class ChatMessage(BaseModel):
    """Повідомлення в діалозі з агентом."""
    
    role: str = Field(..., description="Роль відправника: 'user' або 'assistant'")
    content: str = Field(..., description="Текст повідомлення")


class AgentChatRequest(BaseModel):
    """Запит до діалогового агента з підтримкою сесій."""
    
    session_id: str = Field(default="default_session", description="Унікальний ідентифікатор сесії чату")
    message: str = Field(..., description="Нове повідомлення від користувача")


class AgentChatResponse(BaseModel):
    """Відповідь від діалогового агента."""
    
    session_id: str = Field(..., description="Ідентифікатор сесії чату")
    answer: str = Field(..., description="Підсумкова відповідь агента")
    used_tools: List[str] = Field(default_factory=list, description="Перелік використаних тулзів під час генерації")


class SessionHistoryResponse(BaseModel):
    """Відповідь із повною історією повідомлень сесії."""
    session_id: str
    messages: List[ChatMessage]


class GraphState(TypedDict):
    """
    Контейнер стану (State), який передається між вузлами (Nodes) графа LangGraph.
    """

    query: str
    collection_name: str

    context: str
    documents: List[Document]
    retrieved_docs: List[Document]

    generation: Optional[str]
    answer: Optional[str]

    is_safe: bool
    rejection_reason: Optional[str]
    source_files: List[str]