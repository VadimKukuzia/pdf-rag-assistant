import traceback
from langchain_core.tools import tool
from graph import run_rag_pipeline


@tool
def search_pdf_documents(query: str) -> str:
    """
    Шукає специфічну інформацію, факти, деталі або роз'яснення у завантажених PDF-документах.
    """
    try:
        result = run_rag_pipeline(query)

        # 1. Пробуємо взяти context з результату
        context = result.get("context", "")

        # 2. Якщо context порожній, але є чанки в documents / retrieved_docs — збираємо їх
        if not context or not context.strip():
            docs = result.get("documents") or result.get("retrieved_docs") or []
            if docs:
                context = "\n\n".join([doc.page_content for doc in docs if hasattr(doc, "page_content")])

        answer = result.get("answer") or result.get("generation", "")

        if not context or not context.strip():
            return "У завантажених PDF-документах не знайдено релевантної інформації. Перевірте, чи був завантажений документ."

        return f"Знайдений контекст у PDF:\n{context}\n\nСформована витяжка: {answer}"
    except Exception as e:
        print("\n❌ Помилка всередині search_pdf_documents:")
        traceback.print_exc()
        return f"Помилка виконання RAG-пайплайну: {str(e)}"