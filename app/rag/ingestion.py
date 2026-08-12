import re
from pathlib import Path
from typing import Dict, Any, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.rag.bm25_cache import invalidate_bm25
from app.core.config import settings, policy

CHROMA_PATH = "./chroma_db"


def clean_pdf_text(text: str) -> str:
    text = re.sub(r'(?<![.:;•!?])\n(?![•\n])', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return re.sub(r'[ \t]+', ' ', text).strip()


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.gemini_api_key
    )


def get_vectorstore(collection_name: str = "docs") -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH
    )


def get_all_documents(collection_name: str = "docs") -> List[Document]:
    vectorstore = get_vectorstore(collection_name=collection_name)
    data = vectorstore.get()

    documents = []
    if data and data.get("documents"):
        docs = data["documents"]
        metadatas = data.get("metadatas") or [{}] * len(docs)
        for text, metadata in zip(docs, metadatas):
            if text:
                documents.append(Document(page_content=text, metadata=metadata or {}))
    return documents


def ingest_pdf(file_path: str, collection_name: str = "docs") -> Dict[str, Any]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл {file_path} не знайдено.")

    allowed_exts = policy.get("upload_policy", {}).get("allowed_extensions", [".pdf"])
    if path.suffix.lower() not in allowed_exts:
        raise ValueError(f"Непідтримуване розширення '{path.suffix}'. Дозволені: {allowed_exts}")

    max_mb = policy.get("upload_policy", {}).get("max_file_size_mb", 10)
    if path.stat().st_size > max_mb * 1024 * 1024:
        raise ValueError(f"Файл занадто великий. Максимальний розмір: {max_mb} MB")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    if not documents:
        raise ValueError("PDF-файл порожній або з нього не вдалося зчитати текст.")

    for doc in documents:
        doc.page_content = clean_pdf_text(doc.page_content)

    rag_settings = policy.get("rag_settings", {})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag_settings.get("chunk_size", 1000),
        chunk_overlap=rag_settings.get("chunk_overlap", 200),
        separators=["\n\n", ". ", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source_file"] = path.name

    vectorstore = get_vectorstore(collection_name=collection_name)
    vectorstore.add_documents(chunks)
    
    invalidate_bm25(collection_name)

    return {
        "status": "success",
        "filename": path.name,
        "chunks_created": len(chunks),
        "collection_name": collection_name
    }