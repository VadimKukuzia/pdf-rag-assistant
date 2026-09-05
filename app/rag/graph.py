from typing import Any

from langgraph.graph import END, StateGraph

from app.model.schemas import GraphState
from app.rag.nodes import (
    generator_node,
    guard_node,
    rephraser_node,
    retriever_node,
    validator_node,
)


def check_safety_router(state: GraphState) -> str:
    return "safe" if state.get("is_safe", True) else "unsafe"


def create_rag_graph():
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


def run_rag_pipeline(query: str, session_id: str = "default_session") -> dict[str, Any]:
    initial_state: GraphState = {
        "query": query,
        "collection_name": "docs",
        "context": "",
        "documents": [],
        "retrieved_docs": [],
        "generation": "",
        "answer": None,
        "is_safe": True,
        "rejection_reason": None,
        "source_files": []
    }

    config = {"configurable": {"thread_id": session_id}}
    return app_graph.invoke(initial_state, config=config)