import argparse
import os

from enterprise_rag_agent.agents.rag_agent import create_rag_agent
from enterprise_rag_agent.graph.workflow import build_workflow
from enterprise_rag_agent.rag.embeddings import BedrockEmbeddingService
from enterprise_rag_agent.rag.llm import BedrockLLMService
from enterprise_rag_agent.rag.pipeline import process_pdf
from enterprise_rag_agent.rag.retriever import SemanticRetriever
from enterprise_rag_agent.rag.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("question")
    args = parser.parse_args()

    region = os.getenv("AWS_REGION", "us-east-1")

    embeddings = BedrockEmbeddingService(region=region)
    store = VectorStore(dimension=1024)
    retriever = SemanticRetriever(embeddings, store)

    chunks = process_pdf(args.pdf_path)
    if not chunks:
        raise ValueError("No extractable text found in PDF")

    print(f"Indexing {len(chunks)} chunks...")
    retriever.index_chunks(chunks)

    agent = create_rag_agent(retriever)
    llm = BedrockLLMService(region=region)

    def document_handler(question: str) -> str:
        return str(agent(question))

    def general_handler(question: str) -> str:
        return llm.generate_general(question)

    workflow = build_workflow(
        document_handler=document_handler,
        general_handler=general_handler,
    )

    result = workflow.invoke({"question": args.question})

    print(f"\nROUTE: {result['route']}")
    print(f"\nANSWER:\n{result['answer']}")


if __name__ == "__main__":
    main()
