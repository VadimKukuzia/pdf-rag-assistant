from typing import Any
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain.retrievers.ensemble import EnsembleRetriever

from config import settings
from ingestion import get_vectorstore, get_all_documents


def get_hybrid_retriever(
    collection_name: str = "docs",
    top_k: int = None,
    dense_weight: float = None,
    sparse_weight: float = None
) -> Any:
    """
    Створює гібридний ретривер (EnsembleRetriever), що поєднує:
    - Dense Search (ChromaDB + Gemini Embeddings)
    - Sparse Search (BM25 per-keyword search)
    """
    k = top_k or settings.hybrid_top_k
    d_weight = dense_weight if dense_weight is not None else settings.dense_weight
    s_weight = sparse_weight if sparse_weight is not None else settings.sparse_weight

    vectorstore = get_vectorstore(collection_name=collection_name)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    documents = get_all_documents(collection_name=collection_name)

    # Якщо база порожня, повертаємо звичайний Dense Retriever
    if not documents:
        return dense_retriever

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[d_weight, s_weight]
    )

    return hybrid_retriever