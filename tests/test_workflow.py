
from unittest.mock import Mock

from enterprise_rag_agent.graph.workflow import build_workflow


def test_general_route_does_not_call_document_agent():
    document_handler = Mock(return_value="Document answer")
    general_handler = Mock(return_value="General answer")

    workflow = build_workflow(
        document_handler=document_handler,
        general_handler=general_handler,
    )

    result = workflow.invoke({
        "question": "What is a Python decorator?"
    })

    assert result["route"] == "general"
    assert result["answer"] == "General answer"

    general_handler.assert_called_once()
    document_handler.assert_not_called()

from enterprise_rag_agent.graph.workflow import (
    classify_question,
)


def test_document_routing():
    assert classify_question(
        "Summarize the uploaded PDF"
    ) == "document"


def test_general_routing():
    assert classify_question(
        "What is AWS Lambda?"
    ) == "general"


def test_document_workflow():
    workflow = build_workflow(
        document_handler=lambda q: "Document answer",
        general_handler=lambda q: "General answer",
    )

    result = workflow.invoke({
        "question": "According to the document, what is EC2?"
    })

    assert result["route"] == "document"
    assert result["answer"] == "Document answer"


def test_general_workflow():
    workflow = build_workflow(
        document_handler=lambda q: "Document answer",
        general_handler=lambda q: "General answer",
    )

    result = workflow.invoke({
        "question": "Explain Python decorators"
    })

    assert result["route"] == "general"
    assert result["answer"] == "General answer"
