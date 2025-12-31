"""
Cloudflare R2 upload helper for Eve API.
Handles music library, dreams, memories, and knowledge uploads.
"""
import os
import sys
import logging

logger = logging.getLogger(__name__)

# Ensure root is on path to import shared client
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Check if boto3 is available
BOTO3_AVAILABLE = False
try:
    import boto3
    import botocore
    BOTO3_AVAILABLE = True
    logger.info(f"✅ boto3 {boto3.__version__} and botocore {botocore.__version__} available")
except ImportError as ie:
    logger.error(f"❌ CRITICAL: boto3 not installed: {ie}")
    logger.error(f"❌ This is blocking R2 functionality")
    logger.error(f"❌ FIX: Rebuild Docker with updated requirements.txt")
    logger.error(f"❌ Command: docker-compose build eve")

try:
    from cloudflare_r2_client import R2Client
    R2_AVAILABLE = True
    logger.info("✅ R2Client imported successfully")
except ImportError as ie:
    R2_AVAILABLE = False
    logger.error(f"❌ R2Client import failed: {ie}")
except Exception as e:
    R2_AVAILABLE = False
    logger.error(f"❌ Unexpected error importing R2Client: {type(e).__name__}: {e}")

_r2_client = None

def get_r2_client():
    """Get or create the R2 client from environment variables."""
    global _r2_client
    
    if not BOTO3_AVAILABLE:
        logger.error("❌ CRITICAL: boto3 is not available - cannot initialize R2 client")
        logger.error("❌ The Docker image was not rebuilt with boto3 in requirements.txt")
        logger.error("❌ SOLUTION: docker-compose build eve && docker-compose up eve")
        return None
    
    if not R2_AVAILABLE:
        logger.warning("❌ R2_AVAILABLE is False - R2Client import failed")
        return None
        
    if _r2_client is not None:
        return _r2_client if _r2_client is not False else None
    
    endpoint = os.getenv("R2_ENDPOINT", "").strip()
    access = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    secret = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    default_bucket = os.getenv("R2_DEFAULT_BUCKET", "eve-creations").strip() or "eve-creations"
    
    if not (endpoint and access and secret):
        logger.error(f"❌ R2 credentials incomplete: endpoint={bool(endpoint)}, access={bool(access)}, secret={bool(secret)}")
        _r2_client = False
        return None
    
    try:
        logger.info(f"🔧 Creating R2Client...")
        logger.debug(f"   endpoint: {endpoint}")
        logger.debug(f"   access_key: {access[:10]}...")
        logger.debug(f"   secret_key: {secret[:10]}...")
        logger.debug(f"   bucket: {default_bucket}")
        
        _r2_client = R2Client(endpoint, access, secret, default_bucket)
        logger.info(f"✅ R2 Client created successfully")
        return _r2_client
    except Exception as e:
        logger.error(f"❌ R2Client init FAILED: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
        _r2_client = False
        return None
        return None
    except Exception as e:
        logger.error(f"❌ R2Client init failed: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        _r2_client = False
        return None

def upload_music_to_r2(local_path, key=None, bucket=None):
    """Upload a music file to R2 and return presigned URL."""
    client = get_r2_client()
    if not client or not os.path.isfile(local_path):
        return None
    
    try:
        bucket = bucket or os.getenv("R2_MUSIC_BUCKET") or os.getenv("R2_DEFAULT_BUCKET")
        result = client.upload_file(
            local_path,
            key=key or os.path.basename(local_path),
            bucket=bucket,
            cache_control="public, max-age=31536000"
        )
        url = client.generate_presigned_url(result["key"], bucket=result["bucket"], expires_in_seconds=86400)
        return {"bucket": result["bucket"], "key": result["key"], "presigned_url": url}
    except Exception:
        return None

def upload_dream_to_r2(local_path, key=None, bucket=None):
    """Upload a dream artifact (image/signature) to R2."""
    client = get_r2_client()
    if not client or not os.path.isfile(local_path):
        return None
    
    try:
        bucket = bucket or os.getenv("R2_DREAMS_BUCKET") or os.getenv("R2_DEFAULT_BUCKET")
        result = client.upload_file(
            local_path,
            key=key or os.path.basename(local_path),
            bucket=bucket,
            cache_control="public, max-age=31536000"
        )
        url = client.generate_presigned_url(result["key"], bucket=result["bucket"], expires_in_seconds=86400)
        return {"bucket": result["bucket"], "key": result["key"], "presigned_url": url}
    except Exception:
        return None

def upload_memory_to_r2(local_path_or_bytes, key=None, bucket=None):
    """Upload a memory artifact (session log, transcript) to R2. Accepts file path or bytes."""
    client = get_r2_client()
    if not client:
        return None
    
    try:
        bucket = bucket or os.getenv("R2_MEMORIES_BUCKET") or os.getenv("R2_DEFAULT_BUCKET")
        
        # Handle bytes upload
        if isinstance(local_path_or_bytes, bytes):
            result = client.upload_bytes(
                local_path_or_bytes,
                key=key or "session.json",
                bucket=bucket,
                content_type="application/json",
                cache_control="private, max-age=3600"
            )
        else:
            # Handle file path upload
            if not os.path.isfile(local_path_or_bytes):
                return None
            result = client.upload_file(
                local_path_or_bytes,
                key=key or os.path.basename(local_path_or_bytes),
                bucket=bucket,
                cache_control="private, max-age=3600"
            )
        
        url = client.generate_presigned_url(result["key"], bucket=result["bucket"], expires_in_seconds=3600)
        return {"bucket": result["bucket"], "key": result["key"], "presigned_url": url}
    except Exception:
        return None

def upload_knowledge_to_r2(local_path, key=None, bucket=None):
    """Upload a knowledge document (PDF, TXT, embedding source) to R2."""
    client = get_r2_client()
    if not client or not os.path.isfile(local_path):
        return None
    
    try:
        bucket = bucket or os.getenv("R2_KNOWLEDGE_BUCKET") or os.getenv("R2_DEFAULT_BUCKET")
        result = client.upload_file(
            local_path,
            key=key or os.path.basename(local_path),
            bucket=bucket,
            cache_control="private, max-age=86400"
        )
        url = client.generate_presigned_url(result["key"], bucket=result["bucket"], expires_in_seconds=3600)
        return {"bucket": result["bucket"], "key": result["key"], "presigned_url": url}
    except Exception:
        return None
