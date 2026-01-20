import hashlib
import os
import pickle
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")

# Cache TTL (Time To Live) in seconds - default 7 days
# Can be overridden via CACHE_TTL_DAYS environment variable
DEFAULT_TTL_DAYS = 7
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_DAYS", DEFAULT_TTL_DAYS)) * 24 * 60 * 60

def get_file_hash(file_path):
    """Generate MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_cache_path(file_hash):
    """Get the cache file path for a given hash."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{file_hash}.pkl"


def is_cache_expired(cache_path):
    """
    Check if a cache file has expired based on TTL.

    Args:
        cache_path: Path to the cache file

    Returns:
        True if expired or doesn't exist, False if still valid
    """
    if not cache_path.exists():
        return True

    # Get file modification time
    file_mtime = cache_path.stat().st_mtime
    current_time = time.time()
    age_seconds = current_time - file_mtime

    if age_seconds > CACHE_TTL_SECONDS:
        days_old = age_seconds / (24 * 60 * 60)
        logger.info(f"Cache expired: {cache_path.name} is {days_old:.1f} days old (TTL: {CACHE_TTL_SECONDS // (24*60*60)} days)")
        return True

    return False


def get_cache_age_days(cache_path):
    """Get the age of a cache file in days."""
    if not cache_path.exists():
        return None
    file_mtime = cache_path.stat().st_mtime
    age_seconds = time.time() - file_mtime
    return age_seconds / (24 * 60 * 60)

def load_from_cache(file_hash):
    """
    Load cached data for a file hash.

    Returns:
        Tuple of (chunks, index) if cached, None otherwise
    """
    cache_path = get_cache_path(file_hash)
    if cache_path.exists():
        logger.debug(f"Cache hit for hash: {file_hash[:8]}...")
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    logger.debug(f"Cache miss for hash: {file_hash[:8]}...")
    return None

def save_to_cache(file_hash, chunks, index):
    """
    Save chunks and index to cache.

    Args:
        file_hash: MD5 hash of the source file
        chunks: List of text chunks
        index: FAISS index (we save the vectors, not the index object)
    """
    cache_path = get_cache_path(file_hash)
    logger.info(f"Saving to cache: {file_hash[:8]}... ({len(chunks)} chunks)")
    try:
        # Store chunks and reconstruct index on load
        import faiss
        import numpy as np

        # Extract vectors from index
        vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d)
        vectors = np.array(vectors).reshape(index.ntotal, index.d).copy()

        with open(cache_path, "wb") as f:
            pickle.dump({
                "chunks": chunks,
                "vectors": vectors
            }, f)
        logger.info(f"Cache saved successfully")
    except Exception as e:
        # Caching failure is not critical
        logger.warning(f"Failed to cache: {e}")

def load_cached_index(file_hash):
    """
    Load cached chunks and rebuild FAISS index.
    Returns (None, None) if cache doesn't exist or has expired.

    Returns:
        Tuple of (chunks, index) if cached and valid, (None, None) otherwise
    """
    cache_path = get_cache_path(file_hash)

    # Check if cache exists
    if not cache_path.exists():
        logger.debug(f"No cache found for hash: {file_hash[:8]}...")
        return None, None

    # Check if cache has expired
    if is_cache_expired(cache_path):
        logger.info(f"Cache expired for hash: {file_hash[:8]}..., will reprocess")
        # Optionally delete the expired cache file
        try:
            cache_path.unlink()
            logger.debug(f"Deleted expired cache file: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Failed to delete expired cache: {e}")
        return None, None

    # Cache exists and is valid, load it
    logger.info(f"Loading cached index for hash: {file_hash[:8]}...")
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        import faiss
        import numpy as np

        chunks = data["chunks"]
        vectors = np.asarray(data["vectors"], dtype=np.float32)

        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)

        age_days = get_cache_age_days(cache_path)
        logger.info(f"Cache loaded: {len(chunks)} chunks, {vectors.shape[0]} vectors (age: {age_days:.1f} days)")
        return chunks, index
    except Exception as e:
        logger.warning(f"Failed to load cached index: {e}")
        return None, None

def clear_cache():
    """Clear all cached data."""
    if CACHE_DIR.exists():
        count = 0
        for f in CACHE_DIR.glob("*.pkl"):
            f.unlink()
            count += 1
        logger.info(f"Cache cleared: {count} files removed")


def clear_expired_cache():
    """
    Clear only expired cache files.

    Returns:
        Number of expired files removed
    """
    if not CACHE_DIR.exists():
        return 0

    count = 0
    for cache_file in CACHE_DIR.glob("*.pkl"):
        if is_cache_expired(cache_file):
            try:
                cache_file.unlink()
                count += 1
                logger.debug(f"Removed expired cache: {cache_file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove expired cache {cache_file.name}: {e}")

    if count > 0:
        logger.info(f"Expired cache cleanup: {count} files removed")
    return count


def get_cache_stats():
    """
    Get statistics about the cache.

    Returns:
        Dict with cache statistics
    """
    if not CACHE_DIR.exists():
        return {
            "total_files": 0,
            "total_size_mb": 0,
            "expired_files": 0,
            "valid_files": 0
        }

    total_files = 0
    total_size = 0
    expired_files = 0
    valid_files = 0

    for cache_file in CACHE_DIR.glob("*.pkl"):
        total_files += 1
        total_size += cache_file.stat().st_size

        if is_cache_expired(cache_file):
            expired_files += 1
        else:
            valid_files += 1

    return {
        "total_files": total_files,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "expired_files": expired_files,
        "valid_files": valid_files,
        "ttl_days": CACHE_TTL_SECONDS // (24 * 60 * 60)
    }
