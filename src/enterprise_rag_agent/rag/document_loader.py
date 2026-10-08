from pathlib import Path

from pypdf import PdfReader


def load_pdf(file_path: str) -> list[dict]:
    """Extract text from each page of a PDF."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("File must be a PDF")

    reader = PdfReader(path)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append({
                "content": text.strip(),
                "source": path.name,
                "page": page_number,
            })

    return documents
