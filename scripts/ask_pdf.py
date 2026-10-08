import argparse
import os

from enterprise_rag_agent.rag.embeddings import BedrockEmbeddingService
from enterprise_rag_agent.rag.llm import BedrockLLMService
from enterprise_rag_agent.rag.pipeline import process_pdf
from enterprise_rag_agent.rag.rag_chain import RAGChain
from enterprise_rag_agent.rag.retriever import SemanticRetriever
from enterprise_rag_agent.rag.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("question")
    args = parser.parse_args()

    region = os.getenv("AWS_REGION", "us-east-1")

    embeddings = BedrockEmbeddingService(region=region)
    vector_store = VectorStore(dimension=1024)
    retriever = SemanticRetriever(embeddings, vector_store)
    llm = BedrockLLMService(region=region)

    print("Loading and chunking PDF...")
    chunks = process_pdf(args.pdf_path)

    if not chunks:
        raise ValueError("No extractable text found in PDF")

    print(f"Indexing {len(chunks)} chunks...")
    retriever.index_chunks(chunks)

    rag = RAGChain(retriever, llm)

    print("Generating answer...")
    result = rag.ask(args.question)

    print("\nANSWER\n")
    print(result["answer"])

    print("\nRETRIEVED SOURCES")
    for source in result["sources"]:
        print(
            f"- {source['source']} "
            f"(page {source['page']}, "
            f"score {source['score']:.3f})"
        )


if __name__ == "__main__":
    main()
