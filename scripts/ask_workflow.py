import argparse
import hashlib
import os
from pathlib import Path

from botocore.exceptions import ClientError

from enterprise_rag_agent.agents.rag_agent import create_rag_agent
from enterprise_rag_agent.graph.workflow import build_workflow
from enterprise_rag_agent.rag.embeddings import BedrockEmbeddingService
from enterprise_rag_agent.rag.llm import BedrockLLMService
from enterprise_rag_agent.rag.pipeline import process_pdf
from enterprise_rag_agent.rag.retriever import SemanticRetriever
from enterprise_rag_agent.rag.vector_store import VectorStore
from enterprise_rag_agent.storage.s3_index import S3IndexStorage


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

    
    bucket = os.getenv("RAG_S3_BUCKET")

    if not bucket:
        raise ValueError("RAG_S3_BUCKET environment variable is required")

    s3_storage = S3IndexStorage(bucket=bucket)

    index_file = index_dir / "index.faiss"
    metadata_file = index_dir / "metadata.json"

    if index_file.is_file() and metadata_file.is_file():
        print("Loading existing local FAISS index...")
        store.load(str(index_dir))

    else:
        # Check whether the index already exists in S3.
        s3_key = f"{s3_storage.prefix}/{cache_key}/index.faiss"

        try:
            s3_storage.s3.head_object(
                Bucket=bucket,
                Key=s3_key,
            )
            s3_exists = True

        except ClientError as error:
            error_code = error.response["Error"]["Code"]

            if error_code in ("404", "NoSuchKey", "NotFound"):
                s3_exists = False
            else:
                raise

        if s3_exists:
            print("Downloading existing FAISS index from S3...")
            s3_storage.download(str(index_dir), cache_key)
            store.load(str(index_dir))
            print("FAISS index loaded from S3.")

        else:
            print("No existing index found. Building FAISS index...")

            chunks = process_pdf(str(pdf_path))

            if not chunks:
                raise ValueError("No extractable text found in PDF")

            print(f"Indexing {len(chunks)} chunks...")
            retriever.index_chunks(chunks)

            store.save(str(index_dir))
            print("FAISS index saved locally.")

            s3_storage.upload(str(index_dir), cache_key)
            print("FAISS index uploaded to S3.")



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
