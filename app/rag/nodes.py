from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings, policy
from app.model.schemas import GraphState
from app.rag.ingestion import get_vectorstore

from app.rag.retriever import get_hybrid_retriever


def get_llm() -> ChatGoogleGenerativeAI:
    """Повертає ініціалізований об'єкт Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.gemini_api_key,
        temperature=0
    )


def guard_node(state: GraphState) -> Dict[str, Any]:
    """Перевіряє запит на prompt injection та відповідність суворим лімітам довжини."""
    query = state["query"]
    blocked_keywords = policy["security"]["prompt_injection"]["blocked_keywords"]
    max_length = policy["security"]["prompt_injection"]["max_prompt_length"]

    if len(query) > max_length:
        return {
            "is_safe": False,
            "rejection_reason": f"Перевищено максимальну довжину запиту ({max_length} символів).",
            "generation": policy["responses"]["fallback_security_rejection"]
        }

    query_lower = query.lower()
    for kw in blocked_keywords:
        if kw.lower() in query_lower:
            return {
                "is_safe": False,
                "rejection_reason": f"Виявлено небезпечну фразу: '{kw}'",
                "generation": policy["responses"]["fallback_security_rejection"]
            }

    return {"is_safe": True, "rejection_reason": None}


def rephraser_node(state: GraphState) -> Dict[str, Any]:
    """Переформульовує запит користувача для покращення якості векторного пошуку."""
    if not state.get("is_safe", True):
        return {}

    query = state["query"]
    llm = get_llm()

    system_prompt = (
        "Ти — модуль очищення запитів для пошукової RAG-системи.\n"
    "Твоє єдине завдання — зробити запит користувача чітким та автономним, розкривши займенники з історії бесіди (якщо вони є).\n\n"
    "СУВОРІ ПРАВИЛА:\n"
    "1. НЕ ДОДАВАЙ загальних термінів, наукових штампів чи слів, яких не було у запиті (наприклад: 'методи комп'ютерного зору', 'теорія обробки', 'етапи даних').\n"
    "2. Зберігай оригінальне формулювання та термінологію користувача якомога точніше.\n"
    "3. Якщо запит і так чіткий і не містить відсилань до минулих повідомлень, ПОВЕРНИ ЙОГО БЕЗ ЗМІН.\n\n"

    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ])
        
        rephrased = str(response.content).strip()
        return {"query": rephrased if rephrased else query}
    except Exception:
        return {"query": query}


def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Нода для гібридного пошуку контексту."""
    question = state.get("query") or state.get("question", "")
    collection_name = state.get("collection_name", "docs")

    hybrid_retriever = get_hybrid_retriever(collection_name=collection_name)
    docs = hybrid_retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source_file") or doc.metadata.get("source", "документ.pdf") for doc in docs]))
    
    return {
        **state,
        "context": context,
        "documents": docs,
        "retrieved_docs": docs,
        "source_files": sources  # Заповнюємо список джерел
    }

def generator_node(state: GraphState) -> Dict[str, Any]:
    if not state.get("is_safe", True):
        return {}

    docs = state.get("documents", [])
    query = state["query"]

    if not docs:
        return {"generation": policy["responses"]["fallback_no_context"]}

    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])

    system_prompt = (
        "You are an official AI consultant. Answer the user's question STRICTLY based on the provided context.\n"
        "If the context does not contain the answer, state that clearly in Ukrainian.\n"
        "Do NOT hallucinate or add outside knowledge.\n"
        "Answer in Ukrainian.\n\n"
        f"PROVIDED CONTEXT:\n{context_text}"
    )

    try:
        llm = get_llm()
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Запитання: {query}")
        ])

        raw_content = response.content
        clean_text = ""

        if isinstance(raw_content, str):
            clean_text = raw_content
        elif isinstance(raw_content, list):
            extracted_parts = []
            for part in raw_content:
                if isinstance(part, dict) and "text" in part:
                    extracted_parts.append(part["text"])
                elif isinstance(part, str):
                    extracted_parts.append(part)
            clean_text = "".join(extracted_parts)
        elif isinstance(raw_content, dict) and "text" in raw_content:
            clean_text = raw_content["text"]
        else:
            clean_text = str(raw_content)

        return {"generation": clean_text.strip()}

    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            return {
                "generation": "⚠️ Тимчасово вичерпано квоту Gemini API (429). Будь ласка, зачекайте 30-60 секунд і спробуйте ще раз."
            }
        raise e

def validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Перевіряє, чи згенерована відповідь повністю відповідає знайденому контексту."""
    if not state.get("is_safe", True):
        return state

    generation = state.get("generation") or state.get("answer", "")
    
    # Підтримуємо обидва варіанти ключів для документів
    docs = state.get("documents") or state.get("retrieved_docs") or []
    
    # Якщо context вже є в state — беремо його, інакше збираємо з docs
    context_text = state.get("context") or "\n".join([doc.page_content for doc in docs if hasattr(doc, "page_content")])

    fallback_response = policy["responses"]["fallback_no_context"]

    if not context_text.strip() or not generation or generation == fallback_response:
        return {
            "generation": fallback_response,
            "answer": fallback_response,
            "context": "",
            "is_safe": False
        }

    system_prompt = (
        "Ти — контролер якості RAG-відповідей. Оціни, чи згенерована відповідь базується "
        "СУВОРО на наданому контексті і чи немає в ній вигаданих фактів (галюцинацій).\n"
        "Відповідай ТІЛЬКИ одним словом: 'ТАК' (якщо відповідь правдива і з контексту) "
        "або 'НІ' (якщо є вигадки)."
    )

    user_prompt = f"КОНТЕКСТ:\n{context_text}\n\nЗГЕНЕРОВАНА ВІДПОВІДЬ:\n{generation}"

    try:
        llm = get_llm()
        res = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        eval_result = str(res.content).strip().upper()
        if "НІ" in eval_result or "NO" in eval_result:
            return {
                "generation": fallback_response,
                "answer": fallback_response,
                "context": context_text,
                "is_safe": False
            }
    except Exception as e:
        print(f"⚠️ Помилка під час валідації: {e}")

    # Явно повертаємо оновлені ключі, щоб LangSmith бачив Output і context не втрачався
    return {
        "generation": generation,
        "answer": generation,
        "context": context_text,
        "is_safe": True
    }