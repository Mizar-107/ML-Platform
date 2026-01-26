"""S3 storage operations."""

from typing import Iterator, BinaryIO
import structlog

import boto3
from botocore.exceptions import ClientError

logger = structlog.get_logger()


class S3Client:
    """Client for S3 storage operations."""

    def __init__(
        self,
        bucket: str,
        region: str = "us-west-2",
        prefix: str = "",
    ):
        self.bucket = bucket
        self.region = region
        self.prefix = prefix
        self.client = boto3.client("s3", region_name=region)

    def upload_file(
        self,
        file_path: str,
        key: str,
        metadata: dict | None = None,
    ) -> str:
        """Upload file to S3.

        Args:
            file_path: Local file path
            key: S3 object key
            metadata: Optional metadata

        Returns:
            S3 URI of uploaded object
        """
        full_key = f"{self.prefix}/{key}" if self.prefix else key
        extra_args = {"Metadata": metadata} if metadata else {}

        self.client.upload_file(
            file_path,
            self.bucket,
            full_key,
            ExtraArgs=extra_args,
        )

        uri = f"s3://{self.bucket}/{full_key}"
        logger.info("Uploaded file", uri=uri)
        return uri

    def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: str = "application/octet-stream",
        metadata: dict | None = None,
    ) -> str:
        """Upload bytes to S3.

        Args:
            data: Bytes to upload
            key: S3 object key
            content_type: Content type
            metadata: Optional metadata

        Returns:
            S3 URI of uploaded object
        """
        full_key = f"{self.prefix}/{key}" if self.prefix else key

        self.client.put_object(
            Bucket=self.bucket,
            Key=full_key,
            Body=data,
            ContentType=content_type,
            Metadata=metadata or {},
        )

        uri = f"s3://{self.bucket}/{full_key}"
        logger.info("Uploaded bytes", uri=uri, size=len(data))
        return uri

    def download_file(self, key: str, file_path: str) -> str:
        """Download file from S3.

        Args:
            key: S3 object key
            file_path: Local file path to save

        Returns:
            Local file path
        """
        full_key = f"{self.prefix}/{key}" if self.prefix else key

        self.client.download_file(self.bucket, full_key, file_path)

        logger.info("Downloaded file", key=full_key, path=file_path)
        return file_path

    def download_bytes(self, key: str) -> bytes:
        """Download bytes from S3.

        Args:
            key: S3 object key

        Returns:
            Downloaded bytes
        """
        full_key = f"{self.prefix}/{key}" if self.prefix else key

        response = self.client.get_object(Bucket=self.bucket, Key=full_key)
        data = response["Body"].read()

        logger.debug("Downloaded bytes", key=full_key, size=len(data))
        return data

    def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> Iterator[dict]:
        """List objects in bucket.

        Args:
            prefix: Key prefix filter
            max_keys: Maximum keys per page

        Yields:
            Object metadata dicts
        """
        full_prefix = f"{self.prefix}/{prefix}" if self.prefix else prefix

        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=full_prefix,
            PaginationConfig={"PageSize": max_keys},
        ):
            for obj in page.get("Contents", []):
                yield {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"],
                    "etag": obj["ETag"],
                }

    def delete_object(self, key: str) -> bool:
        """Delete object from S3.

        Args:
            key: S3 object key

        Returns:
            True if deleted
        """
        full_key = f"{self.prefix}/{key}" if self.prefix else key

        try:
            self.client.delete_object(Bucket=self.bucket, Key=full_key)
            logger.info("Deleted object", key=full_key)
            return True
        except ClientError as e:
            logger.error("Failed to delete object", key=full_key, error=str(e))
            return False

    def exists(self, key: str) -> bool:
        """Check if object exists.

        Args:
            key: S3 object key

        Returns:
            True if exists
        """
        full_key = f"{self.prefix}/{key}" if self.prefix else key

        try:
            self.client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except ClientError:
            return False

    def generate_presigned_url(
        self,
        key: str,
        expiration: int = 3600,
        operation: str = "get_object",
    ) -> str:
        """Generate presigned URL for object.

        Args:
            key: S3 object key
            expiration: URL expiration in seconds
            operation: S3 operation (get_object, put_object)

        Returns:
            Presigned URL
        """
        full_key = f"{self.prefix}/{key}" if self.prefix else key

        url = self.client.generate_presigned_url(
            operation,
            Params={"Bucket": self.bucket, "Key": full_key},
            ExpiresIn=expiration,
        )

        return url


def create_s3_client(
    bucket: str,
    region: str = "us-west-2",
    prefix: str = "",
) -> S3Client:
    """Create S3 client.

    Args:
        bucket: S3 bucket name
        region: AWS region
        prefix: Optional key prefix

    Returns:
        S3Client instance
    """
    return S3Client(bucket=bucket, region=region, prefix=prefix)
