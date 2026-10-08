
import json
import os
from pathlib import Path

from enterprise_rag_agent.rag.embeddings import BedrockEmbeddingService
from enterprise_rag_agent.rag.pipeline import process_pdf
from enterprise_rag_agent.rag.retriever import SemanticRetriever
from enterprise_rag_agent.rag.vector_store import VectorStore


def main():
    dataset_path = Path("evaluation/dataset.json")
    pdf_path = "data/sample/aws-overview.pdf"

    dataset = json.loads(dataset_path.read_text())

    region = os.getenv("AWS_REGION", "us-east-1")

    embeddings = BedrockEmbeddingService(region=region)
    store = VectorStore(dimension=1024)
    retriever = SemanticRetriever(embeddings, store)

    chunks = process_pdf(pdf_path)
    print(f"Indexing {len(chunks)} chunks...")
    retriever.index_chunks(chunks)

    successful = 0

    for item in dataset:
        results = retriever.retrieve(item["question"], k=5)

        retrieved_text = " ".join(
            result["content"] for result in results
        ).lower()

        matched = [
            keyword
            for keyword in item["expected_keywords"]
            if keyword.lower() in retrieved_text
        ]

        score = len(matched) / len(item["expected_keywords"])

        successful += score >= 0.5

        print(f"\nQuestion: {item['question']}")
        print(f"Keyword coverage: {score:.2f}")
        print(f"Matched: {matched}")

    pass_rate = successful / len(dataset)

    print(f"\nRetrieval smoke-test pass rate: {pass_rate:.2%}")


if __name__ == "__main__":
    main()
