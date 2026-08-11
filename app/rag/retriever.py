from typing import Any
from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers import EnsembleRetriever

from app.core.config import settings
from app.rag.ingestion import get_vectorstore, get_all_documents


def get_hybrid_retriever(
    collection_name: str = "docs",
    top_k: int = None,
    dense_weight: float = None,
    sparse_weight: float = None
) -> Any:
    k = top_k or getattr(settings, "hybrid_top_k", 4)
    d_weight = dense_weight if dense_weight is not None else getattr(settings, "dense_weight", 0.5)
    s_weight = sparse_weight if sparse_weight is not None else getattr(settings, "sparse_weight", 0.5)

    vectorstore = get_vectorstore(collection_name=collection_name)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    documents = get_all_documents(collection_name=collection_name)
    if not documents:
        return dense_retriever

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[d_weight, s_weight]
    )