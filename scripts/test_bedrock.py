import os

from enterprise_rag_agent.rag.embeddings import (
    BedrockEmbeddingService,
)


def main():
    service = BedrockEmbeddingService(
        region=os.getenv("AWS_REGION", "us-east-1")
    )

    embedding = service.embed_text(
        "Amazon Bedrock provides managed foundation models."
    )

    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First five values: {embedding[:5]}")


if __name__ == "__main__":
    main()
