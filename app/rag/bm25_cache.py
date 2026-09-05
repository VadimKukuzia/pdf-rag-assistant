from langchain_community.retrievers import BM25Retriever

_bm25_cache: dict[str, BM25Retriever] = {}


def get_cached_bm25(collection_name: str) -> BM25Retriever | None:
    return _bm25_cache.get(collection_name)


def set_cached_bm25(
    collection_name: str,
    retriever: BM25Retriever,
) -> None:
    _bm25_cache[collection_name] = retriever


def invalidate_bm25(collection_name: str = "docs") -> None:
    _bm25_cache.pop(collection_name, None)