import io
import json

import pytest

from enterprise_rag_agent.rag.embeddings import (
    BedrockEmbeddingService,
)


class FakeBedrockClient:
    def invoke_model(self, **kwargs):
        self.request = kwargs

        return {
            "body": io.BytesIO(
                json.dumps({
                    "embedding": [0.1, 0.2, 0.3],
                }).encode()
            )
        }


def test_embedding_generation():
    client = FakeBedrockClient()
    service = BedrockEmbeddingService(client=client)

    embedding = service.embed_text("Amazon Bedrock")

    assert embedding == [0.1, 0.2, 0.3]
    assert client.request["modelId"] == (
        "amazon.titan-embed-text-v2:0"
    )


def test_empty_text():
    service = BedrockEmbeddingService(
        client=FakeBedrockClient()
    )

    with pytest.raises(ValueError):
        service.embed_text("")


def test_embed_chunks():
    service = BedrockEmbeddingService(
        client=FakeBedrockClient()
    )

    chunks = [{
        "content": "Amazon Bedrock provides AI services.",
        "source": "aws.pdf",
        "page": 1,
        "chunk_id": "aws.pdf-p1-c0",
    }]

    result = service.embed_chunks(chunks)

    assert len(result) == 1
    assert result[0]["embedding"] == [0.1, 0.2, 0.3]
    assert result[0]["source"] == "aws.pdf"
