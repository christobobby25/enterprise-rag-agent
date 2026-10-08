import json

import boto3


class BedrockEmbeddingService:
    def __init__(self, region="us-east-1", client=None):
        self.client = client or boto3.client(
            "bedrock-runtime",
            region_name=region,
        )

        self.model_id = "amazon.titan-embed-text-v2:0"

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Text cannot be empty")

        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "inputText": text,
                "dimensions": 1024,
                "normalize": True,
            }),
        )

        result = json.loads(response["body"].read())
        return result["embedding"]

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        embedded_chunks = []

        for chunk in chunks:
            embedding = self.embed_text(chunk["content"])

            embedded_chunks.append({
                **chunk,
                "embedding": embedding,
            })

        return embedded_chunks
