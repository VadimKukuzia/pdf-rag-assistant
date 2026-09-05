import logging

from langchain_core.tools import tool

from app.rag.graph import run_rag_pipeline

logger = logging.getLogger(__name__)


@tool
def search_pdf_documents(query: str) -> str:
    """Шукає специфічну інформацію, факти, деталі або роз'яснення у завантажених PDF-документах."""
    try:
        result = run_rag_pipeline(query)
        if not result.get("is_safe", True):
            return result.get(
                "generation",
                "Запит відхилено системою безпеки."
            )
        context = result.get("context", "")

        if not context or not context.strip():
            docs = result.get("documents") or result.get("retrieved_docs") or []
            context = "\n\n".join([doc.page_content for doc in docs if hasattr(doc, "page_content")])

        answer = result.get("answer") or result.get("generation", "")

        if not context or not context.strip():
            return "У завантажених PDF-документах не знайдено релевантної інформації. Перевірте, чи був завантажений документ."

        return f"Знайдений контекст у PDF:\n{context}\n\nСформована витяжка: {answer}"
    except Exception as e:
        logger.error("Error inside search_pdf_documents tool: %s", e, exc_info=True)
        return f"Помилка виконання RAG-пайплайну: {e!s}"