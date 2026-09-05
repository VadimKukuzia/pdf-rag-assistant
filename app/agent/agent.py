import logging
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agent.tools import search_pdf_documents
from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ви — ввічливий, розумний та професійний AI-асистент.\n"
    "Ваша мета — вести природну та корисну бесіду з користувачем.\n\n"
    "Правила використання інструментів:\n"
    "1. На звичайні привітання, прості питання або загальну розмову відповідайте самостійно БЕЗ виклику інструментів.\n"
    "2. Якщо запитання стосується змісту завантажених PDF-документів або конкретних фактів з файлів, "
    "ОБОВ'ЯЗКОВО викликайте інструмент `search_pdf_documents`.\n"
    "3. Формуйте фінальну відповідь зрозуміло, спираючись на дані з інструменту."
)


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part if isinstance(part, str) else part.get("text", "")
            for part in content
            if isinstance(part, (str, dict))
        ]
        return "\n".join(filter(None, parts))
    return str(content)


def get_agent():
    llm = ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.gemini_api_key,
        temperature=0.3
    )
    tools = [search_pdf_documents]
    
    try:
        return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    except TypeError:
        return create_agent(model=llm, tools=tools, state_modifier=SYSTEM_PROMPT)


agent_executor = get_agent()


def run_agent_chat(message: str, history: list[dict[str, str]] = None) -> dict[str, Any]:
    messages = []
    for msg in (history or []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=message))

    response = agent_executor.invoke({"messages": messages})
    all_messages = response.get("messages", [])

    final_message = _extract_text_content(all_messages[-1].content) if all_messages else ""

    used_tools = [
        tc.get("name", "unknown_tool")
        for msg in all_messages
        if hasattr(msg, "tool_calls") and msg.tool_calls
        for tc in msg.tool_calls
    ]

    return {
        "answer": final_message,
        "used_tools": list(set(used_tools))
    }