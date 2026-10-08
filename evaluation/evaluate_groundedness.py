import json
import os

from enterprise_rag_agent.evaluation.groundedness import (
    GroundednessEvaluator,
)
from enterprise_rag_agent.rag.embeddings import BedrockEmbeddingService
from enterprise_rag_agent.rag.llm import BedrockLLMService
from enterprise_rag_agent.rag.pipeline import process_pdf
from enterprise_rag_agent.rag.retriever import SemanticRetriever
from enterprise_rag_agent.rag.vector_store import VectorStore


def main():
    region = os.getenv("AWS_REGION", "us-east-1")

    with open("evaluation/dataset.json", encoding="utf-8") as file:
        dataset = json.load(file)

    embeddings = BedrockEmbeddingService(region=region)
    store = VectorStore(dimension=1024)
    retriever = SemanticRetriever(embeddings, store)
    llm = BedrockLLMService(region=region)
    judge = GroundednessEvaluator(region=region)

    chunks = process_pdf("data/sample/aws-overview.pdf")
    print(f"Indexing {len(chunks)} chunks...")
    retriever.index_chunks(chunks)

    scores = []

    for item in dataset:
        question = item["question"]
        results = retriever.retrieve(question, k=5)

        context = "\n\n".join(
            result["content"] for result in results
        )

        answer = llm.generate(
            question=question,
            context=context,
        )

        evaluation = judge.evaluate(
            question=question,
            answer=answer,
            context=context,
        )

        scores.append(evaluation["score"])

        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")
        print(f"Groundedness: {evaluation['score']}/5")
        print(f"Explanation: {evaluation['explanation']}")
        print(
            "Unsupported claims:",
            evaluation["unsupported_claims"],
        )

    average = sum(scores) / len(scores)
    print(f"\nAverage groundedness: {average:.2f}/5")


if __name__ == "__main__":
    main()
