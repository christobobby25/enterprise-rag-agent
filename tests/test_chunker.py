import pytest

from enterprise_rag_agent.rag.chunker import chunk_documents


def test_small_document():
    documents = [
        {
            "content": "Amazon Bedrock supports foundation models.",
            "source": "aws.pdf",
            "page": 1,
        }
    ]

    chunks = chunk_documents(documents)

    assert len(chunks) == 1
    assert chunks[0]["source"] == "aws.pdf"
    assert chunks[0]["page"] == 1
    assert chunks[0]["chunk_id"] == "aws.pdf-p1-c0"


def test_large_document():
    documents = [
        {
            "content": "Amazon Bedrock provides AI services. " * 100,
            "source": "aws.pdf",
            "page": 2,
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=200,
        chunk_overlap=40,
    )

    assert len(chunks) > 1
    assert all(len(chunk["content"]) <= 200 for chunk in chunks)
    assert all(chunk["page"] == 2 for chunk in chunks)


def test_invalid_chunk_settings():
    with pytest.raises(ValueError):
        chunk_documents([], chunk_size=100, chunk_overlap=100)


def test_empty_documents():
    assert chunk_documents([]) == []
