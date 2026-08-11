import datetime
from sqlalchemy import String, Text, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base


class ChatSessionModel(Base):
    """Модель сесії чату."""
    __tablename__ = "chat_sessions"

    session_id: Mapped[str] = mapped_column(String(255), primary_key=True, index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    messages: Mapped[list["ChatMessageModel"]] = relationship(
        "ChatMessageModel", back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessageModel(Base):
    """Модель окремого повідомлення в історії чату."""
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.session_id"))
    role: Mapped[str] = mapped_column(String(50))  # 'user' або 'assistant'
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    session: Mapped["ChatSessionModel"] = relationship("ChatSessionModel", back_populates="messages")