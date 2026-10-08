
import pytest

from enterprise_rag_agent.rag.vector_store import VectorStore


def test_save_and_load_vector_store(tmp_path):
    store = VectorStore(dimension=3)

    store.add_documents([
        {
            "content": "Amazon EC2 provides compute capacity.",
            "source": "aws.pdf",
            "page": 1,
            "chunk_id": "chunk-1",
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "content": "Amazon S3 provides object storage.",
            "source": "aws.pdf",
            "page": 2,
            "chunk_id": "chunk-2",
            "embedding": [0.0, 1.0, 0.0],
        },
    ])

    directory = tmp_path / "faiss_index"
    store.save(str(directory))

    assert (directory / "index.faiss").exists()
    assert (directory / "metadata.json").exists()

    restored = VectorStore(dimension=3)
    restored.load(str(directory))

    assert restored.index.ntotal == 2
    assert len(restored.documents) == 2

    results = restored.search([1.0, 0.0, 0.0], k=1)

    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["score"] == pytest.approx(1.0)


def test_load_wrong_dimension(tmp_path):
    store = VectorStore(dimension=3)
    store.add_documents([
        {
            "content": "Test document",
            "embedding": [1.0, 0.0, 0.0],
        }
    ])

    directory = tmp_path / "faiss_index"
    store.save(str(directory))

    restored = VectorStore(dimension=4)

    with pytest.raises(ValueError, match="dimension"):
        restored.load(str(directory))
