from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

try:
    from langchain.agents import create_agent
except ImportError:
    from langgraph.prebuilt import create_react_agent as create_agent

from app.core.config import settings
from app.agent.tools import search_pdf_documents

SYSTEM_PROMPT = (
    "Ви — ввічливий, розумний та професійний AI-асистент.\n"
    "Ваша мета — вести природну та корисну бесіду з користувачем.\n\n"
    "Правила використання інструментів:\n"
    "1. На звичайні привітання, прості питання або загальну розмову відповідайте самостійно БЕЗ виклику інструментів.\n"
    "2. Якщо запитання стосується змісту завантажених PDF-документів, конкретних фактів чи матеріалів з файлів, "
    "ОБОВ'ЯЗКОВО викликайте інструмент `search_pdf_documents`.\n"
    "3. Формуйте фінальну відповідь зрозуміло, спираючись на дані, повернуті інструментом."
)


def _extract_text_content(content: Any) -> str:
    """Нормалізує content з AI-повідомлення у чисту строку."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "\n".join(text_parts)
    return str(content)


def get_agent():
    """Ініціалізує Conversational Tool-Calling Агента."""
    llm = ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.gemini_api_key,
        temperature=0.3
    )
    
    tools = [search_pdf_documents]
    
    try:
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT
        )
    except TypeError:
        agent = create_agent(
            model=llm,
            tools=tools,
            state_modifier=SYSTEM_PROMPT
        )
        
    return agent


agent_executor = get_agent()


def run_agent_chat(
    message: str,
    history: List[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Приймає повідомлення користувача та історію діалогу, повертає відповідь."""
    history = history or []
    messages = []

    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=message))

    response = agent_executor.invoke({"messages": messages})
    
    all_messages = response.get("messages", [])
    
    if all_messages:
        raw_content = all_messages[-1].content
        final_message = _extract_text_content(raw_content)
    else:
        final_message = ""

    used_tools = []
    for msg in all_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                used_tools.append(tc.get("name", "unknown_tool"))

    return {
        "answer": final_message,
        "used_tools": list(set(used_tools))
    }