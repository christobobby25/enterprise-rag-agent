from unittest.mock import MagicMock

import agentcore_app


def test_valid_invocation(monkeypatch):
    mock_workflow = MagicMock()

    mock_workflow.invoke.return_value = {
        "route": "document",
        "answer": "Amazon EC2 provides cloud compute capacity.",
    }

    monkeypatch.setattr(
        agentcore_app,
        "_workflow",
        mock_workflow,
    )

    result = agentcore_app.invoke(
        {"question": "What is Amazon EC2?"}
    )

    assert result["route"] == "document"
    assert "Amazon EC2" in result["answer"]

    mock_workflow.invoke.assert_called_once_with(
        {"question": "What is Amazon EC2?"}
    )


def test_empty_question():
    result = agentcore_app.invoke({"question": ""})

    assert result == {
        "error": "A non-empty question is required"
    }


def test_missing_question():
    result = agentcore_app.invoke({})

    assert "error" in result
