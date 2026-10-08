from enterprise_rag_agent.rag.llm import BedrockLLMService
from enterprise_rag_agent.rag.retriever import SemanticRetriever


class RAGChain:
    def __init__(
        self,
        retriever: SemanticRetriever,
        llm: BedrockLLMService,
    ):
        self.retriever = retriever
        self.llm = llm

    def ask(self, question: str, k: int = 3) -> dict:
        if not question.strip():
            raise ValueError("Question cannot be empty")

        documents = self.retriever.retrieve(question, k=k)

        if not documents:
            return {
                "answer": "No relevant documents are indexed.",
                "sources": [],
            }

        context_parts = []

        for index, doc in enumerate(documents, start=1):
            context_parts.append(
                f"[{index}] Source: {doc['source']}, "
                f"Page: {doc['page']}\n"
                f"{doc['content']}"
            )

        context = "\n\n".join(context_parts)
        answer = self.llm.generate(question, context)

        sources = [
            {
                "source": doc["source"],
                "page": doc["page"],
                "chunk_id": doc["chunk_id"],
                "score": doc["score"],
            }
            for doc in documents
        ]

        return {
            "answer": answer,
            "sources": sources,
        }
