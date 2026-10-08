
from collections.abc import Callable
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class RAGState(TypedDict, total=False):
    question: str
    route: Literal["document", "general"]
    answer: str


def classify_question(question: str) -> Literal["document", "general"]:
    """Initial rule-based router; replace with LLM routing later."""
    keywords = (
        "document",
        "pdf",
        "according to",
        "in the file",
        "in the report",
        "uploaded",
        "summarize",
    )

    if any(word in question.lower() for word in keywords):
        return "document"

    return "general"


def build_workflow(
    document_handler: Callable[[str], str],
    general_handler: Callable[[str], str],
):
    """Build a LangGraph workflow with two answer paths."""

    def router(state: RAGState) -> dict:
        return {"route": classify_question(state["question"])}

    def document_node(state: RAGState) -> dict:
        return {"answer": document_handler(state["question"])}

    def general_node(state: RAGState) -> dict:
        return {"answer": general_handler(state["question"])}

    graph = StateGraph(RAGState)

    graph.add_node("router", router)
    graph.add_node("document", document_node)
    graph.add_node("general", general_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "document": "document",
            "general": "general",
        },
    )

    graph.add_edge("document", END)
    graph.add_edge("general", END)

    return graph.compile()
