import pytest

from enterprise_rag_agent.rag.retriever import SemanticRetriever
from enterprise_rag_agent.rag.vector_store import VectorStore


class FakeEmbeddingService:
    def embed_text(self, text):
        if "Lambda" in text or "serverless" in text.lower():
            return [1.0, 0.0, 0.0]

        return [0.0, 1.0, 0.0]

    def embed_chunks(self, chunks):
        return [
            {
                **chunk,
                "embedding": self.embed_text(chunk["content"]),
            }
            for chunk in chunks
        ]


def test_semantic_retrieval():
    store = VectorStore(dimension=3)

    retriever = SemanticRetriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
    )

    chunks = [
        {
            "content": "AWS Lambda automatically scales.",
            "source": "aws.pdf",
            "page": 1,
            "chunk_id": "aws.pdf-p1-c0",
        },
        {
            "content": "Amazon S3 stores objects.",
            "source": "aws.pdf",
            "page": 2,
            "chunk_id": "aws.pdf-p2-c0",
        },
    ]

    retriever.index_chunks(chunks)

    results = retriever.retrieve(
        "How does serverless scaling work?",
        k=1,
    )

    assert len(results) == 1
    assert results[0]["page"] == 1
    assert results[0]["score"] == pytest.approx(1.0)


def test_empty_vector_store():
    store = VectorStore(dimension=3)

    assert store.search([1.0, 0.0, 0.0]) == []


def test_invalid_embedding_dimension():
    store = VectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.add_documents([
            {
                "content": "Example",
                "embedding": [1.0, 0.0],
            }
        ])


def test_empty_question():
    retriever = SemanticRetriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=VectorStore(dimension=3),
    )

    with pytest.raises(ValueError):
        retriever.retrieve("")
