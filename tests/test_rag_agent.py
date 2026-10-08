from unittest.mock import Mock, patch

from enterprise_rag_agent.agents.rag_agent import create_rag_agent


@patch("enterprise_rag_agent.agents.rag_agent.Agent")
@patch("enterprise_rag_agent.agents.rag_agent.BedrockModel")
def test_create_rag_agent(mock_model, mock_agent):
    retriever = Mock()

    create_rag_agent(retriever)

    mock_model.assert_called_once()
    mock_agent.assert_called_once()

    kwargs = mock_agent.call_args.kwargs

    assert len(kwargs["tools"]) == 1
    assert "search_documents" in kwargs["system_prompt"]
