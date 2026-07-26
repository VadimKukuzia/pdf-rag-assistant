from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings, policy
from schemas import GraphState
from ingestion import get_vectorstore


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
        "Ти — експерт із семантичного пошуку. Твоє завдання — переформулювати запит користувача "
        "так, щоб він містив чіткі ключові слова для векторного пошуку в документації. "
        "Збережи суть запитання. Відповідай ВИКЛЮЧНО переформульованим запитом без вступу та лапок."
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


def retriever_node(state: GraphState) -> Dict[str, Any]:
    """Шукає найрелевантніші чанки документів у векторній базі ChromaDB."""
    if not state.get("is_safe", True):
        return {"documents": [], "source_files": []}

    query = state["query"]
    top_k = policy["rag_settings"]["top_k_results"]

    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=top_k)

    source_files = list(set([doc.metadata.get("source_file", "невідомо") for doc in docs]))

    return {
        "documents": docs,
        "source_files": source_files
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

def validator_node(state: GraphState) -> Dict[str, Any]:
    """Перевіряє, чи згенерована відповідь повністю відповідає знайденому контексту."""
    if not state.get("is_safe", True):
        return {}

    generation = state.get("generation", "")
    docs = state.get("documents", [])

    if not docs or not generation or generation == policy["responses"]["fallback_no_context"]:
        return {"generation": policy["responses"]["fallback_no_context"]}

    context_text = "\n".join([doc.page_content for doc in docs])

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
            return {"generation": policy["responses"]["fallback_no_context"]}
    except Exception:
        pass

    return {}