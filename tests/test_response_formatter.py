import pytest

from enterprise_rag_agent.agents.response_formatter import clean_agent_response


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            ("<thinking>Searching documents.</thinking>"
             "<response>The answer is AWS.</response>"),
            "The answer is AWS.",
        ),
        (
            "<thinking>\nStep 1\nStep 2\n</thinking>\nFinal answer.",
            "Final answer.",
        ),
        (
            "AWS provides cloud computing services.",
            "AWS provides cloud computing services.",
        ),
        (
            ("<THINKING>Internal reasoning</THINKING>"
             "<RESPONSE>Final answer</RESPONSE>"),
            "Final answer",
        ),
    ],
)
def test_clean_agent_response(raw, expected):
    assert clean_agent_response(raw) == expected
