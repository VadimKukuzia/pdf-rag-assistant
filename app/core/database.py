import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

# Використовує PostgreSQL, якщо задано в .env, або за замовчуванням асинхронний SQLite для локальної розробки
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./chat_history.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Ініціалізація схем та таблиць бази даних."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Залежність (Dependency) для отримання сесії БД у FastAPI."""
    async with AsyncSessionLocal() as session:
        yield session