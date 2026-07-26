from typing import Dict, Any
from langgraph.graph import StateGraph, END

from schemas import GraphState
from nodes import (
    guard_node,
    rephraser_node,
    retriever_node,
    generator_node,
    validator_node
)


def check_safety_router(state: GraphState) -> str:
    """
    Визначає наступний крок після Guard-ноди.
    
    Якщо запит безпечний -> йдемо на переформулювання та пошук.
    Якщо виявлено injection або порушення лімітів -> миттєво завершуємо граф.
    """
    if state.get("is_safe", True):
        return "safe"
    return "unsafe"


def create_rag_graph() -> StateGraph:
    """Створює та налаштовує топологію графа LangGraph."""
    
    workflow = StateGraph(GraphState)

    workflow.add_node("guard", guard_node)
    workflow.add_node("rephraser", rephraser_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("validator", validator_node)

    workflow.set_entry_point("guard")

    workflow.add_conditional_edges(
        "guard",
        check_safety_router,
        {
            "safe": "rephraser",
            "unsafe": END
        }
    )

    workflow.add_edge("rephraser", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "validator")
    workflow.add_edge("validator", END)

    return workflow.compile()


app_graph = create_rag_graph()


def run_rag_pipeline(query: str, session_id: str = "default_session") -> Dict[str, Any]:
    """
    Функція-обгортка для запуску графа з початковим станом.
    """
    initial_state: GraphState = {
        "query": query,
        "documents": [],
        "generation": None,
        "is_safe": True,
        "rejection_reason": None,
        "source_files": []
    }

    config = {"configurable": {"thread_id": session_id}}
    final_state = app_graph.invoke(initial_state, config=config)

    return final_state