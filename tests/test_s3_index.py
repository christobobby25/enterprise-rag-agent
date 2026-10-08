
from unittest.mock import MagicMock

import pytest

from enterprise_rag_agent.storage.s3_index import S3IndexStorage


def test_upload_index(tmp_path):
    (tmp_path / "index.faiss").write_bytes(b"fake-index")
    (tmp_path / "metadata.json").write_text("{}")

    mock_s3 = MagicMock()
    storage = S3IndexStorage(
        bucket="test-bucket",
        s3_client=mock_s3,
    )

    storage.upload(str(tmp_path), "test-key")

    assert mock_s3.upload_file.call_count == 2

    mock_s3.upload_file.assert_any_call(
        str(tmp_path / "index.faiss"),
        "test-bucket",
        "faiss-indexes/test-key/index.faiss",
    )


def test_upload_missing_file(tmp_path):
    mock_s3 = MagicMock()
    storage = S3IndexStorage(
        bucket="test-bucket",
        s3_client=mock_s3,
    )

    with pytest.raises(FileNotFoundError):
        storage.upload(str(tmp_path), "test-key")

    mock_s3.upload_file.assert_not_called()


def test_download_index(tmp_path):
    mock_s3 = MagicMock()
    storage = S3IndexStorage(
        bucket="test-bucket",
        s3_client=mock_s3,
    )

    directory = tmp_path / "downloaded"
    storage.download(str(directory), "test-key")

    assert mock_s3.download_file.call_count == 2

    mock_s3.download_file.assert_any_call(
        "test-bucket",
        "faiss-indexes/test-key/index.faiss",
        str(directory / "index.faiss"),
    )
