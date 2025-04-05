import os
import threading

import httpx
import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()

_lock = threading.Lock()
_sync_client = None
_async_client = None


def get_openai_client(is_async: bool = True) -> OpenAI | AsyncOpenAI:
    """Get OpenAI client instance.

    Args:
        is_async: Whether to return async client. Defaults to True.

    Returns:
        OpenAI or AsyncOpenAI client instance
    """
    global _sync_client, _async_client

    with _lock:
        # Check if the API key is set
        api_key = openai.api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "OpenAI API key is not set. Please set it using set_openai_key."
            )

        if is_async:
            if _async_client is None:
                _async_client = AsyncOpenAI(
                    api_key=api_key,
                    timeout=httpx.Timeout(60.0, read=40, connect=5.0),
                    max_retries=10,
                )
            return _async_client
        else:
            if _sync_client is None:
                _sync_client = OpenAI(
                    api_key=api_key,
                    timeout=httpx.Timeout(60.0, read=40, connect=5.0),
                    max_retries=10,
                )
            return _sync_client


def set_openai_client(new_client: OpenAI | AsyncOpenAI, is_async: bool = False):
    """Set OpenAI client instance.

    Args:
        new_client: New client instance to use
        is_async: Whether the client is async. Defaults to False.
    """
    global _sync_client, _async_client
    with _lock:
        if is_async:
            _async_client = new_client
        else:
            _sync_client = new_client


def set_openai_key(key: str):
    """Set OpenAI API key.

    Args:
        key: API key to use
    """
    if not key:
        raise ValueError("Invalid API key. The API key cannot be empty.")

    openai.api_key = key

    global _sync_client, _async_client
    with _lock:
        _sync_client = None
        _async_client = None
