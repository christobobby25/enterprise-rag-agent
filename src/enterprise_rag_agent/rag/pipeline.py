from enterprise_rag_agent.rag.chunker import chunk_documents
from enterprise_rag_agent.rag.document_loader import load_pdf


def process_pdf(file_path: str) -> list[dict]:
    """Load a PDF and prepare its chunks for retrieval."""

    documents = load_pdf(file_path)

    chunks = chunk_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=200,
    )

    return chunks
