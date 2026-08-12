import logging
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings, policy
from app.model.schemas import GraphState
from app.rag.retriever import get_hybrid_retriever

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.gemini_api_key,
        temperature=temperature
    )


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts)
    if isinstance(content, dict) and "text" in content:
        return content["text"]
    return str(content)


def guard_node(state: GraphState) -> Dict[str, Any]:
    query = state.get("query", "")
    max_length = policy.get("security", {}).get("prompt_injection", {}).get("max_prompt_length", 1000)
    blocked_keywords = policy.get("security", {}).get("prompt_injection", {}).get("blocked_keywords", [])

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
    if not state.get("is_safe", True):
        return {}

    query = state.get("query", "")
    system_prompt = (
        "Ти — модуль очищення запитів для пошукової RAG-системи.\n"
        "Твоє завдання — зробити запит чітким та автономним, розкривши займенники з контексту.\n\n"
        "СУВОРІ ПРАВИЛА:\n"
        "1. НЕ ДОДАВАЙ загальних термінів чи наукових штампів.\n"
        "2. Зберігай оригінальне формулювання користувача.\n"
        "3. Якщо запит чіткий, ПОВЕРНИ ЙОГО БЕЗ ЗМІН."
    )

    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ])
        rephrased = _extract_text(response.content).strip()
        return {"query": rephrased if rephrased else query}
    except Exception as e:
        logger.warning("Rephraser node error, using original query: %s", e)
        return {"query": query}


def retriever_node(state: GraphState) -> Dict[str, Any]:
    query = state.get("query", "")
    collection_name = state.get("collection_name", "docs")

    hybrid_retriever = get_hybrid_retriever(collection_name=collection_name)
    docs = hybrid_retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([
        doc.metadata.get("source_file") or doc.metadata.get("source", "документ.pdf") 
        for doc in docs
    ]))

    return {
        "context": context,
        "documents": docs,
        "retrieved_docs": docs,
        "source_files": sources
    }


def generator_node(state: GraphState) -> Dict[str, Any]:
    if not state.get("is_safe", True):
        return {}

    docs = state.get("documents", [])
    query = state.get("query", "")

    if not docs:
        fallback = policy["responses"]["fallback_no_context"]
        return {"generation": fallback, "answer": fallback}

    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])

    system_prompt = (
        "You are an official AI consultant. Answer the user's question STRICTLY based on the provided context.\n"
        "If the context does not contain the answer, state that clearly in Ukrainian.\n"
        "Do NOT hallucinate or add outside knowledge.\n"
        "Answer in Ukrainian.\n\n"
        f"PROVIDED CONTEXT:\n{context_text}"
    )

    try:
        llm = get_llm(temperature=0.2)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Запитання: {query}")
        ])
        clean_text = _extract_text(response.content).strip()
        return {"generation": clean_text, "answer": clean_text}

    except Exception as e:
        logger.error("Generator node execution error: %s", e, exc_info=True)
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            fallback = "⚠️ Тимчасово вичерпано квоту Gemini API (429). Будь ласка, зачекайте 30-60 секунд."
            return {"generation": fallback, "answer": fallback}
        raise e


def validator_node(state: GraphState) -> Dict[str, Any]:
    if not state.get("is_safe", True):
        return state

    generation = state.get("generation") or state.get("answer", "")
    docs = state.get("documents") or state.get("retrieved_docs") or []
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
        "СУВОРО на наданому контексті і чи немає в ній вигаданих фактів.\n"
        "Відповідай ТІЛЬКИ одним словом: 'ТАК' або 'НІ'."
    )
    user_prompt = f"КОНТЕКСТ:\n{context_text}\n\nЗГЕНЕРОВАНА ВІДПОВІДЬ:\n{generation}"

    try:
        llm = get_llm(temperature=0.0)
        res = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        eval_result = _extract_text(res.content).strip().upper()
        if "НІ" in eval_result or "NO" in eval_result:
            return {
                "generation": fallback_response,
                "answer": fallback_response,
                "context": context_text,
                "is_safe": False
            }
    except Exception as e:
        logger.warning("Validator node check failed: %s", e)

    return {
        "generation": generation,
        "answer": generation,
        "context": context_text,
        "is_safe": True
    }