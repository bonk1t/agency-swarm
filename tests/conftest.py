"""Test configuration for agency_swarm tests."""

from __future__ import annotations

import os
import sys

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment."""
    # Ensure we have the required environment variables
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY must be set in .env file"
