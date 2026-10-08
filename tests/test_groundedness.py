import json
from unittest.mock import Mock

import pytest

from enterprise_rag_agent.evaluation.groundedness import (
    GroundednessEvaluator,
)


@pytest.fixture
def evaluator():
    judge = GroundednessEvaluator.__new__(GroundednessEvaluator)
    judge.client = Mock()
    judge.model_id = "test-model"
    return judge


def mock_response(evaluator, text):
    evaluator.client.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": text}]
            }
        },
        "stopReason": "end_turn",
    }


def test_valid_json(evaluator):
    payload = {
        "score": 5,
        "explanation": "Fully supported",
        "unsupported_claims": [],
    }

    mock_response(evaluator, json.dumps(payload))

    result = evaluator.evaluate("Question", "Answer", "Context")

    assert result["score"] == 5


def test_markdown_wrapped_json(evaluator):
    payload = (
        '```json\n'
        '{"score": 5, "explanation": "Supported", '
        '"unsupported_claims": []}\n'
        '```'
    )

    mock_response(evaluator, payload)

    result = evaluator.evaluate("Question", "Answer", "Context")

    assert result["score"] == 5


def test_invalid_json(evaluator):
    mock_response(evaluator, "This is not JSON")

    with pytest.raises(json.JSONDecodeError):
        evaluator.evaluate("Question", "Answer", "Context")


def test_invalid_score(evaluator):
    payload = {
        "score": 10,
        "explanation": "Invalid score",
        "unsupported_claims": [],
    }

    mock_response(evaluator, json.dumps(payload))

    with pytest.raises(ValueError):
        evaluator.evaluate("Question", "Answer", "Context")
