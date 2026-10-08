
import pytest

from enterprise_rag_agent.rag.document_loader import load_pdf


def test_missing_pdf():
    with pytest.raises(FileNotFoundError):
        load_pdf("nonexistent.pdf")


def test_invalid_file_type(tmp_path):
    file_path = tmp_path / "document.txt"
    file_path.write_text("Hello world")

    with pytest.raises(ValueError):
        load_pdf(str(file_path))


def test_pdf_extraction(tmp_path):
    from pypdf import PdfWriter

    file_path = tmp_path / "sample.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)

    with open(file_path, "wb") as file:
        writer.write(file)

    result = load_pdf(str(file_path))

    # Blank pages should not produce documents.
    assert result == []
