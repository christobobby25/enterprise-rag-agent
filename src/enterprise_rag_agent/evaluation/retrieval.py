def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Measure precision among the top K retrieved chunks."""
    if k <= 0:
        raise ValueError("k must be positive")

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        chunk_id in relevant_ids
        for chunk_id in top_k
    )

    return relevant_count / len(top_k)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = 5,
) -> float:
    """Measure how many known relevant chunks were retrieved."""
    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])

    return len(top_k & relevant_ids) / len(relevant_ids)
