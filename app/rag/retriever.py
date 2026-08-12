from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from app.core.config import settings
from app.rag.ingestion import get_vectorstore, get_all_documents
from app.rag.bm25_cache import (
    get_cached_bm25,
    set_cached_bm25,
)


def get_bm25_retriever(
    collection_name: str = "docs",
    top_k: int = 4,
) -> BM25Retriever | None:

    retriever = get_cached_bm25(collection_name)

    if retriever is not None:
        retriever.k = top_k
        return retriever

    documents = get_all_documents(collection_name=collection_name)

    if not documents:
        return None

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = top_k

    set_cached_bm25(collection_name, retriever)

    return retriever


def get_hybrid_retriever(
    collection_name: str = "docs",
    top_k: int | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
) -> Any:

    k = top_k or getattr(settings, "hybrid_top_k", 4)

    d_weight = (
        dense_weight
        if dense_weight is not None
        else getattr(settings, "dense_weight", 0.5)
    )

    s_weight = (
        sparse_weight
        if sparse_weight is not None
        else getattr(settings, "sparse_weight", 0.5)
    )

    vectorstore = get_vectorstore(collection_name=collection_name)

    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    bm25_retriever = get_bm25_retriever(
        collection_name=collection_name,
        top_k=k,
    )

    if bm25_retriever is None:
        return dense_retriever

    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[d_weight, s_weight],
    )