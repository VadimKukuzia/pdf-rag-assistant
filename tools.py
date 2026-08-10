from langchain_core.tools import tool
from graph import run_rag_pipeline


@tool
def search_pdf_documents(query: str) -> str:
    """
    Шукає специфічну інформацію, факти, деталі або роз'яснення у завантажених PDF-документах.

    
    :param query: Пошуковий запит для RAG-пайплайну.
    """
    try:
        result = run_rag_pipeline(question=query)
        context = result.get("context", "")
        answer = result.get("answer", "")

        if not context:
            return "У завантажених PDF-документах не знайдено релевантної інформації за вашим запитом."

        return f"Знайдений контекст у PDF:\n{context}\n\nСформована витяжка: {answer}"
    except Exception as e:
        return f"Помилка виконання пошуку в PDF: {str(e)}"