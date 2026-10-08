from enterprise_rag_agent.rag.embeddings import (
    BedrockEmbeddingService,
)
from enterprise_rag_agent.rag.vector_store import VectorStore


class SemanticRetriever:
    def __init__(
        self,
        embedding_service: BedrockEmbeddingService,
        vector_store: VectorStore,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def index_chunks(self, chunks: list[dict]):
        if not chunks:
            return

        embedded_chunks = self.embedding_service.embed_chunks(chunks)
        self.vector_store.add_documents(embedded_chunks)

    def retrieve(self, question: str, k: int = 3):
        if not question.strip():
            raise ValueError("Question cannot be empty")

        query_embedding = self.embedding_service.embed_text(question)

        return self.vector_store.search(
            query_embedding,
            k=k,
        )
