import os
from pathlib import Path

from bedrock_agentcore import BedrockAgentCoreApp

from enterprise_rag_agent.agents.rag_agent import create_rag_agent
from enterprise_rag_agent.agents.response_formatter import clean_agent_response
from enterprise_rag_agent.graph.workflow import build_workflow
from enterprise_rag_agent.rag.embeddings import BedrockEmbeddingService
from enterprise_rag_agent.rag.llm import BedrockLLMService
from enterprise_rag_agent.rag.retriever import SemanticRetriever
from enterprise_rag_agent.rag.vector_store import VectorStore
from enterprise_rag_agent.storage.s3_index import S3IndexStorage

app = BedrockAgentCoreApp()

_workflow = None


def create_workflow():
    """Initialize the RAG workflow using an S3-backed index."""
    region = os.getenv("AWS_REGION", "us-east-1")
    bucket = os.environ["RAG_S3_BUCKET"]
    cache_key = os.environ["RAG_INDEX_KEY"]

    embeddings = BedrockEmbeddingService(region=region)
    store = VectorStore(dimension=1024)
    retriever = SemanticRetriever(embeddings, store)

    index_dir = Path("/tmp/enterprise-rag-index") / cache_key

    storage = S3IndexStorage(bucket=bucket)

    if not (
        (index_dir / "index.faiss").is_file()
        and (index_dir / "metadata.json").is_file()
    ):
        storage.download(str(index_dir), cache_key)

    store.load(str(index_dir))

    document_agent = create_rag_agent(retriever)
    llm = BedrockLLMService(region=region)

    def document_handler(question: str) -> str:
        response = document_agent(question)
        return clean_agent_response(str(response))

    def general_handler(question: str) -> str:
        return llm.generate_general(question)

    return build_workflow(
        document_handler=document_handler,
        general_handler=general_handler,
    )


@app.entrypoint
def invoke(payload: dict) -> dict:
    """Handle a question through the existing LangGraph workflow."""
    global _workflow

    question = payload.get("question")

    if not isinstance(question, str) or not question.strip():
        return {"error": "A non-empty question is required"}

    if _workflow is None:
        _workflow = create_workflow()

    result = _workflow.invoke({"question": question})

    return {
        "question": question,
        "route": result["route"],
        "answer": result["answer"],
    }


if __name__ == "__main__":
    app.run()
