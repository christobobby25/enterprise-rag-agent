import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """In-memory FAISS vector store using cosine similarity."""

    def __init__(self, dimension: int = 1024):
        if dimension <= 0:
            raise ValueError("Dimension must be positive")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    def add_documents(self, embedded_chunks: list[dict]):
        if not embedded_chunks:
            return

        vectors = np.asarray(
            [chunk["embedding"] for chunk in embedded_chunks],
            dtype=np.float32,
        )

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings must contain finite values")

        vectors = np.ascontiguousarray(vectors)
        faiss.normalize_L2(vectors)

        self.index.add(vectors)

        for chunk in embedded_chunks:
            self.documents.append({
                key: value
                for key, value in chunk.items()
                if key != "embedding"
            })

    def search(self, query_embedding: list[float], k: int = 3):
        if k <= 0:
            raise ValueError("k must be positive")

        if self.index.ntotal == 0:
            return []

        query = np.asarray(
            [query_embedding],
            dtype=np.float32,
        )

        if query.shape != (1, self.dimension):
            raise ValueError("Query embedding dimension mismatch")

        if not np.isfinite(query).all():
            raise ValueError("Query must contain finite values")

        if np.linalg.norm(query) == 0:
            raise ValueError("Query embedding cannot be zero")

        query = np.ascontiguousarray(query)
        faiss.normalize_L2(query)

        scores, indices = self.index.search(
            query,
            min(k, self.index.ntotal),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            results.append({
                **self.documents[index],
                "score": float(score),
            })

        return results
    
    def save(self, directory: str) -> None:
        """Persist the FAISS index and document metadata."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(path / "index.faiss"),
        )

        metadata = {
            "dimension": self.dimension,
            "documents": self.documents,
        }

        with (path / "metadata.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=2)

    def load(self, directory: str) -> None:
        """Restore a previously saved FAISS index."""
        path = Path(directory)

        loaded_index = faiss.read_index(
            str(path / "index.faiss"),
        )

        with (path / "metadata.json").open(
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        if loaded_index.d != self.dimension:
            raise ValueError("Saved index dimension mismatch")

        if metadata["dimension"] != self.dimension:
            raise ValueError("Saved metadata dimension mismatch")

        if loaded_index.ntotal != len(metadata["documents"]):
            raise ValueError("Index and document count mismatch")

        self.index = loaded_index
        self.documents = metadata["documents"]

    
