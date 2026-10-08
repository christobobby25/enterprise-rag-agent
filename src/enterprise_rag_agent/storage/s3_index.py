from pathlib import Path

import boto3


class S3IndexStorage:
    """Upload and download persisted FAISS indexes."""

    FILES = ("index.faiss", "metadata.json")

    def __init__(
        self,
        bucket: str,
        prefix: str = "faiss-indexes",
        s3_client=None,
    ):
        if not bucket:
            raise ValueError("S3 bucket is required")

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = s3_client or boto3.client("s3")

    def upload(self, directory: str, cache_key: str) -> None:
        path = Path(directory)

        for filename in self.FILES:
            local_file = path / filename

            if not local_file.is_file():
                raise FileNotFoundError(local_file)

        for filename in self.FILES:
            key = f"{self.prefix}/{cache_key}/{filename}"

            self.s3.upload_file(
                str(path / filename),
                self.bucket,
                key,
            )

    def download(self, directory: str, cache_key: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        for filename in self.FILES:
            key = f"{self.prefix}/{cache_key}/{filename}"

            self.s3.download_file(
                self.bucket,
                key,
                str(path / filename),
            )
