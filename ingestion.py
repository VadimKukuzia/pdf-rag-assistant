from pathlib import Path
from typing import Dict, Any, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings, policy

CHROMA_PATH = "./chroma_db"


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Повертає модель Google для генерації векторних ембедінгів."""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.gemini_api_key
    )


def get_vectorstore(collection_name: str = "docs") -> Chroma:
    """Ініціалізує або підключається до існуючої бази ChromaDB."""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH
    )


def get_all_documents(collection_name: str = "docs") -> List[Document]:
    """Витягує всі документи з ChromaDB для ініціалізації BM25 індексу."""
    vectorstore = get_vectorstore(collection_name=collection_name)
    data = vectorstore.get()

    documents = []
    if data and "documents" in data and data["documents"]:
        for text, metadata in zip(data["documents"], data["metadatas"]):
            documents.append(Document(page_content=text, metadata=metadata or {}))
    return documents


def ingest_pdf(file_path: str, collection_name: str = "docs") -> Dict[str, Any]:
    """
    Приймає шлях до PDF, перевіряє за політиками безпеки,
    нарізає на чанки та зберігає векторні ембедінги в ChromaDB.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл {file_path} не знайдено.")

    if path.suffix.lower() not in policy["upload_policy"]["allowed_extensions"]:
        raise ValueError(
            f"Непідтримуване розширення '{path.suffix}'. "
            f"Дозволені: {policy['upload_policy']['allowed_extensions']}"
        )

    max_bytes = policy["upload_policy"]["max_file_size_mb"] * 1024 * 1024
    if path.stat().st_size > max_bytes:
        raise ValueError(
            f"Файл занадто великий ({path.stat().st_size / (1024*1024):.2f} MB). "
            f"Максимальний дозволений розмір: {policy['upload_policy']['max_file_size_mb']} MB"
        )

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    if not documents:
        raise ValueError("PDF-файл порожній або з нього не вдалося зчитати текст.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=policy["rag_settings"]["chunk_size"],
        chunk_overlap=policy["rag_settings"]["chunk_overlap"],
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source_file"] = path.name

    vectorstore = get_vectorstore(collection_name=collection_name)
    vectorstore.add_documents(chunks)

    return {
        "status": "success",
        "filename": path.name,
        "chunks_created": len(chunks),
        "collection_name": collection_name
    }