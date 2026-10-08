from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """Split PDF pages into overlapping text chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and < chunk_size"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = []

    for document in documents:
        text_chunks = splitter.split_text(document["content"])

        for index, text in enumerate(text_chunks):
            chunks.append(
                {
                    "content": text,
                    "source": document["source"],
                    "page": document["page"],
                    "chunk_id": f"{document['source']}-"
                    f"p{document['page']}-c{index}",
                }
            )

    return chunks
