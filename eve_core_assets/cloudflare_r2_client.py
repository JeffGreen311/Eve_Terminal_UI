import os
import mimetypes
import uuid
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Try to import boto3 - if not available, set placeholders
try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    Config = None
    BotoCoreError = Exception
    ClientError = Exception
    Config = None
    BotoCoreError = Exception
    ClientError = Exception


class R2Client:
    """
    Minimal Cloudflare R2 S3-compatible client.

    Configure via env vars or pass explicitly:
      - R2_ENDPOINT: https://<accountid>.r2.cloudflarestorage.com
      - R2_ACCESS_KEY_ID
      - R2_SECRET_ACCESS_KEY
      - R2_DEFAULT_BUCKET (optional)
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        default_bucket: Optional[str] = None,
        addressing_style: str = "path",
    ) -> None:
        if not endpoint_url or not access_key_id or not secret_access_key:
            raise ValueError("R2 endpoint and credentials are required")
        
        if boto3 is None:
            raise RuntimeError("❌ boto3 is not installed - cannot create R2 client. Run: pip install boto3 botocore")

        try:
            logger.debug(f"🔧 R2Client: Calling boto3.client('s3')...")
            logger.debug(f"   endpoint_url={endpoint_url}")
            logger.debug(f"   region_name=auto")
            logger.debug(f"   addressing_style={addressing_style}")
            
            # For R2, region_name can be 'auto'
            self._s3 = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name="auto",
                config=Config(signature_version="s3v4", s3={"addressing_style": addressing_style}),
            )
            logger.info(f"✅ boto3 S3 client created successfully")
            self.default_bucket = default_bucket
        except BotoCoreError as be:
            logger.error(f"❌ BotoCoreError: {be}")
            raise RuntimeError(f"BotoCoreError: {be}")
        except ClientError as ce:
            logger.error(f"❌ ClientError: {ce}")
            raise RuntimeError(f"ClientError: {ce}")
        except Exception as e:
            logger.error(f"❌ Unexpected error in R2Client.__init__: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"❌ Traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to create boto3 S3 client for R2: {type(e).__name__}: {e}")

    @classmethod
    def from_env(cls) -> "R2Client":
        return cls(
            endpoint_url=os.getenv("R2_ENDPOINT", "").strip(),
            access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            default_bucket=os.getenv("R2_DEFAULT_BUCKET", "").strip() or None,
        )

    def upload_file(
        self,
        local_path: str,
        key: Optional[str] = None,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None,
        cache_control: Optional[str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(local_path)
        key = key or os.path.basename(local_path)
        bucket = bucket or self.default_bucket
        if not bucket:
            raise ValueError("Bucket must be provided (no default set)")

        guessed_ct, _ = mimetypes.guess_type(local_path)
        content_type = content_type or guessed_ct or "application/octet-stream"

        put_args: Dict[str, Any] = {"ContentType": content_type}
        if cache_control:
            put_args["CacheControl"] = cache_control
        if extra_args:
            put_args.update(extra_args)

        try:
            self._s3.upload_file(local_path, bucket, key, ExtraArgs=put_args)
            return {"bucket": bucket, "key": key, "content_type": content_type}
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"R2 upload failed: {e}")

    def upload_bytes(
        self,
        data: bytes,
        key: Optional[str] = None,
        bucket: Optional[str] = None,
        content_type: str = "application/octet-stream",
        cache_control: Optional[str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        key = key or str(uuid.uuid4())
        bucket = bucket or self.default_bucket
        if not bucket:
            raise ValueError("Bucket must be provided (no default set)")

        put_args: Dict[str, Any] = {"ContentType": content_type}
        if cache_control:
            put_args["CacheControl"] = cache_control
        if extra_args:
            put_args.update(extra_args)

        try:
            self._s3.put_object(Bucket=bucket, Key=key, Body=data, **put_args)
            return {"bucket": bucket, "key": key, "content_type": content_type}
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"R2 put_object failed: {e}")

    def generate_presigned_url(
        self,
        key: str,
        bucket: Optional[str] = None,
        expires_in_seconds: int = 3600,
    ) -> str:
        bucket = bucket or self.default_bucket
        if not bucket:
            raise ValueError("Bucket must be provided (no default set)")
        try:
            return self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in_seconds,
            )
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"R2 presign failed: {e}")

    def list_objects(self, bucket: Optional[str] = None, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        bucket = bucket or self.default_bucket
        if not bucket:
            raise ValueError("Bucket must be provided (no default set)")
        try:
            resp = self._s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)
            return resp.get("Contents", [])
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"R2 list_objects failed: {e}")

    def delete_object(self, key: str, bucket: Optional[str] = None) -> None:
        bucket = bucket or self.default_bucket
        if not bucket:
            raise ValueError("Bucket must be provided (no default set)")
        try:
            self._s3.delete_object(Bucket=bucket, Key=key)
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"R2 delete failed: {e}")
