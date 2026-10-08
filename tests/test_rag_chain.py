from unittest.mock import Mock

from enterprise_rag_agent.rag.rag_chain import RAGChain


def test_rag_answer_with_sources():
    retriever = Mock()
    llm = Mock()

    retriever.retrieve.return_value = [
        {
            "content": "AWS Lambda automatically scales.",
            "source": "aws.pdf",
            "page": 2,
            "chunk_id": "aws.pdf-p2-c0",
            "score": 0.95,
        }
    ]

    llm.generate.return_value = (
        "AWS Lambda automatically scales."
    )

    rag = RAGChain(retriever, llm)
    result = rag.ask("How does Lambda scale?")

    assert "automatically scales" in result["answer"]
    assert result["sources"][0]["page"] == 2
    assert result["sources"][0]["chunk_id"] == "aws.pdf-p2-c0"

    llm.generate.assert_called_once()


def test_no_documents():
    retriever = Mock()
    llm = Mock()

    retriever.retrieve.return_value = []

    rag = RAGChain(retriever, llm)
    result = rag.ask("Unknown question")

    assert result["sources"] == []
    assert "No relevant documents" in result["answer"]

    llm.generate.assert_not_called()
