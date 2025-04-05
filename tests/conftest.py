"""Test configuration for agency_swarm tests."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment."""
    # Ensure we have the required environment variables
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY must be set in .env file"
