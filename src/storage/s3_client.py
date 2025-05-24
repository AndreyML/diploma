"""
S3 storage client for handling file uploads and downloads
"""
import boto3
from botocore.exceptions import ClientError
from src.config.settings import settings
from typing import Optional, List
import io
import base64
import logging
logger = logging.getLogger(__name__)
class S3Client:
    """S3 storage client for file operations"""
    def __init__(self):
        """Initialize S3 client"""
        self.client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
            endpoint_url=settings.S3_ENDPOINT_URL
        )
        self.bucket_name = settings.S3_BUCKET_NAME
    def upload_file(self, file_content: bytes, key: str, content_type: str = "application/octet-stream") -> str:
        """
        Upload file to S3
        Args:
            file_content: File content as bytes
            key: S3 object key (path)
            content_type: Content type of the file
        Returns:
            S3 URL of uploaded file
        """
        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=file_content,
                ContentType=content_type
            )
            url = f"https://{self.bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"
            if settings.S3_ENDPOINT_URL:
                url = f"{settings.S3_ENDPOINT_URL}/{self.bucket_name}/{key}"
            logger.info(f"File uploaded to S3: {url}")
            return url
        except ClientError as e:
            logger.error(f"Error uploading file to S3: {e}")
            raise
    def upload_image_from_base64(self, base64_data: str, key: str) -> str:
        """
        Upload image from base64 string to S3
        Args:
            base64_data: Base64 encoded image data
            key: S3 object key (path)
        Returns:
            S3 URL of uploaded image
        """
        try:
            # Decode base64 data
            image_data = base64.b64decode(base64_data)
            # Upload to S3
            return self.upload_file(image_data, key, "image/png")
        except Exception as e:
            logger.error(f"Error uploading image from base64: {e}")
            raise
    def download_file(self, key: str) -> bytes:
        """
        Download file from S3
        Args:
            key: S3 object key (path)
        Returns:
            File content as bytes
        """
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"Error downloading file from S3: {e}")
            raise
    def delete_file(self, key: str) -> bool:
        """
        Delete file from S3
        Args:
            key: S3 object key (path)
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"File deleted from S3: {key}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting file from S3: {e}")
            return False
    def list_files(self, prefix: str = "") -> List[str]:
        """
        List files in S3 bucket with given prefix
        Args:
            prefix: Key prefix to filter files
        Returns:
            List of object keys
        """
        try:
            response = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            logger.error(f"Error listing files from S3: {e}")
            return []
    def file_exists(self, key: str) -> bool:
        """
        Check if file exists in S3
        Args:
            key: S3 object key (path)
        Returns:
            True if file exists, False otherwise
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError:
            return False
# Global S3 client instance
s3_client = S3Client()
