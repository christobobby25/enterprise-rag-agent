import pytest

from enterprise_rag_agent.evaluation.retrieval import (
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    retrieved = ["chunk1", "chunk2", "chunk3", "chunk4"]
    relevant = {"chunk1", "chunk3"}

    assert precision_at_k(retrieved, relevant, k=4) == 0.5


def test_recall_at_k():
    retrieved = ["chunk1", "chunk2", "chunk3"]
    relevant = {"chunk1", "chunk3", "chunk5"}

    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_precision_with_no_results():
    assert precision_at_k([], {"chunk1"}) == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["chunk1"], {"chunk1"}, k=0)
