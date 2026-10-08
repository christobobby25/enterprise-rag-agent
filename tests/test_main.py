
from enterprise_rag_agent.main import get_status


def test_application_status():
    result = get_status()

    assert result["status"] == "running"
    assert result["application"] == "Enterprise RAG Agent"
    assert result["version"] == "0.1.0"
