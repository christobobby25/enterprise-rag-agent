import argparse
import hashlib
import os
from pathlib import Path

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

    
    pdf_path = Path(args.pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Hash the PDF content and embedding configuration.
    pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    embedding_model = "amazon.titan-embed-text-v2:0"
    cache_key = hashlib.sha256(
        f"{pdf_hash}:{embedding_model}:{store.dimension}:"
        "chunk_size=1000:chunk_overlap=200".encode()
    ).hexdigest()[:16]

    index_dir = Path("data/indexes") / cache_key

    if (
        (index_dir / "index.faiss").exists()
        and (index_dir / "metadata.json").exists()
    ):
        print("Loading existing FAISS index...")
        store.load(str(index_dir))
    else:
        print("Building FAISS index...")

        chunks = process_pdf(str(pdf_path))

        if not chunks:
            raise ValueError("No extractable text found in PDF")

        print(f"Indexing {len(chunks)} chunks...")
        retriever.index_chunks(chunks)

        store.save(str(index_dir))
        print("FAISS index saved.")


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
