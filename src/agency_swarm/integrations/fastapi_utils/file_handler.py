import asyncio
import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import aiofiles
import filetype
import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


client = AsyncOpenAI()

logger = logging.getLogger(__name__)


def get_extension_from_name(name: str) -> str | None:
    ext = os.path.splitext(name)[1]
    return ext if ext else None


def get_extension_from_url(url: str) -> str | None:
    path = urlparse(url).path
    ext = os.path.splitext(path)[1]
    return ext if ext else None


def get_extension_from_filetype(file_path: str | Path) -> str | None:
    kind = filetype.guess(str(file_path))
    if kind:
        return f".{kind.extension}"
    return None


async def async_download_file(url: str, name: str, save_dir: str | Path) -> str:
    """
    Helper function to download file from url to local path.
    Args:
        url: The URL of the file to download.
        name: The name of the file to download.
        save_dir: Directory to store the file.
    Returns:
        The local path of the downloaded file.
    """
    # Prioritize user-provided extension
    ext = get_extension_from_name(name) or get_extension_from_url(url)
    base_name = os.path.splitext(name)[0]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    temp_path = Path(save_dir) / f"{base_name}.tmp"
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url, headers=headers) as r:
            r.raise_for_status()
            async with aiofiles.open(temp_path, "wb") as f:
                async for chunk in r.aiter_bytes():
                    await f.write(chunk)
    if not ext:
        ext = get_extension_from_filetype(temp_path)
    if not ext:
        raise ValueError(f"No extension found for file: {url}")
    filename = f"{base_name}{ext}"
    local_path = Path(save_dir) / filename
    if local_path.exists():
        os.remove(local_path)
    os.rename(temp_path, local_path)
    return str(local_path)


async def async_upload_to_openai(file_path: str | Path) -> str:
    with open(file_path, "rb") as f:
        uploaded_file = await client.files.create(file=f, purpose="assistants")
    return uploaded_file.id


async def wait_for_file_ready(file_id: str, max_attempts: int = 30, delay: float = 0.5) -> bool:
    """
    Wait for a file to be ready for use after upload.

    Args:
        file_id: The OpenAI file ID to check
        max_attempts: Maximum number of polling attempts
        delay: Delay between attempts in seconds

    Returns:
        True if file is ready, False if timeout
    """
    for _ in range(max_attempts):
        try:
            file_data = await client.files.retrieve(file_id)
            # Check if file status indicates it's ready (this may vary by API)
            if hasattr(file_data, "status") and file_data.status in ["processed", "ready"]:
                return True
            # If no status field or status is unknown, assume ready after successful retrieval
            if file_data.id == file_id:
                return True
        except Exception as e:
            logger.debug(f"File {file_id} not ready yet: {e}")

        await asyncio.sleep(delay)

    return False


async def upload_from_urls(file_map: dict[str, str]) -> dict[str, str]:
    """
    Helper function to upload files from urls to OpenAI.
    Args:
        file_map: A dictionary mapping file names to URLs.
        in a format of {"file_name": "url", ...}
    Returns:
        A dictionary mapping file names to file IDs.
    """
    file_ids = []
    with tempfile.TemporaryDirectory() as temp_dir:
        download_tasks = [async_download_file(url, name, temp_dir) for name, url in file_map.items()]
        file_paths = await asyncio.gather(*download_tasks)
        upload_tasks = [async_upload_to_openai(path) for path in file_paths]
        file_ids = await asyncio.gather(*upload_tasks)

    # Wait for all files to be ready
    ready_tasks = [wait_for_file_ready(file_id) for file_id in file_ids]
    ready_results = await asyncio.gather(*ready_tasks)

    if not all(ready_results):
        failed_files = [name for name, ready in zip(file_map.keys(), ready_results, strict=True) if not ready]
        logger.warning(f"Some files may not be ready for use: {failed_files}")

    return dict(zip(file_map.keys(), file_ids, strict=True))
