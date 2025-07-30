"""Test file upload attachments without vector stores."""

import asyncio
import os
import shutil
from pathlib import Path

import pytest
from agents import ModelSettings
from openai import AsyncOpenAI

from agency_swarm import Agency, Agent

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@pytest.fixture
async def real_openai_client():
    return AsyncOpenAI(api_key=OPENAI_API_KEY)


@pytest.mark.asyncio
async def test_pdf_attachment_no_vector_store(real_openai_client: AsyncOpenAI, tmp_path: Path):
    """
    Test that PDF files can be attached directly without creating vector stores.
    """
    # Use the test PDF file
    test_pdf_path = Path("tests/data/files/test-pdf.pdf")
    assert test_pdf_path.exists(), f"Test file not found at {test_pdf_path}"

    # Upload file to OpenAI
    with open(test_pdf_path, "rb") as f:
        uploaded_file = await real_openai_client.files.create(file=f, purpose="assistants")

    try:
        # Create an agent without files_folder (no vector store)
        agent = Agent(
            name="PDFReader",
            instructions="You are an agent that can read and analyze PDF files.",
            model_settings=ModelSettings(temperature=0.0),
        )
        agent._openai_client = real_openai_client

        # Initialize agency
        agency = Agency(agent, user_context=None)

        # Ask a question with file attachment
        response_result = await agency.get_response(
            message="What is the total revenue mentioned in this financial report?", file_ids=[uploaded_file.id]
        )

        # Verify response
        assert response_result is not None
        print(f"Response: {response_result.final_output}")

        # The response should contain revenue information
        assert any(term in response_result.final_output.lower() for term in ["revenue", "million", "financial"]), (
            f"Expected revenue information not found in: {response_result.final_output}"
        )

    finally:
        # Cleanup
        try:
            await real_openai_client.files.delete(uploaded_file.id)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_code_file_attachment_no_vector_store(real_openai_client: AsyncOpenAI, tmp_path: Path):
    """
    Test that code files are handled via CodeInterpreter without vector stores.
    """
    # Create a simple Python file
    test_py_path = tmp_path / "test_script.py"
    test_py_path.write_text("""
def calculate_sum(numbers):
    return sum(numbers)

# Test the function
test_numbers = [1, 2, 3, 4, 5]
result = calculate_sum(test_numbers)
print(f"The sum is: {result}")
""")

    # Upload file to OpenAI
    with open(test_py_path, "rb") as f:
        uploaded_file = await real_openai_client.files.create(file=f, purpose="assistants")

    try:
        # Create an agent without files_folder (no vector store)
        agent = Agent(
            name="CodeReader",
            instructions="You are an agent that can read and execute Python code.",
            model_settings=ModelSettings(temperature=0.0),
        )
        agent._openai_client = real_openai_client

        # Initialize agency
        agency = Agency(agent, user_context=None)

        # Ask a question with file attachment
        response_result = await agency.get_response(
            message="Run this Python script and tell me what the result is.", file_ids=[uploaded_file.id]
        )

        # Verify response
        assert response_result is not None
        print(f"Response: {response_result.final_output}")

        # The response should contain the sum result (15)
        assert "15" in response_result.final_output, (
            f"Expected sum result '15' not found in: {response_result.final_output}"
        )

    finally:
        # Cleanup
        try:
            await real_openai_client.files.delete(uploaded_file.id)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_unsupported_file_direct_attachment(real_openai_client: AsyncOpenAI, tmp_path: Path):
    """
    Test that non-PDF, non-code files are attempted as direct attachments with warning.
    """
    # Create a text file (not PDF, not code)
    test_txt_path = tmp_path / "test_document.txt"
    test_txt_path.write_text("This is a test document with some content.")

    # Upload file to OpenAI
    with open(test_txt_path, "rb") as f:
        uploaded_file = await real_openai_client.files.create(file=f, purpose="assistants")

    try:
        # Create an agent without files_folder (no vector store)
        agent = Agent(
            name="TextReader",
            instructions="You are an agent that can read text files.",
            model_settings=ModelSettings(temperature=0.0),
        )
        agent._openai_client = real_openai_client

        # Initialize agency
        agency = Agency(agent, user_context=None)

        # Ask a question with file attachment
        # This should attempt direct attachment and fail with a clear error
        with pytest.raises(Exception) as exc_info:
            await agency.get_response(message="What does this document say?", file_ids=[uploaded_file.id])

        # Verify the error is about unsupported file type
        error_msg = str(exc_info.value)
        assert (
            "Expected file type to be a supported format: .pdf but got .txt" in error_msg
            or "Runner execution failed" in error_msg
        ), f"Unexpected error: {error_msg}"

        print(f"Got expected error for non-PDF file: {error_msg}")

    finally:
        # Cleanup
        try:
            await real_openai_client.files.delete(uploaded_file.id)
        except Exception:
            pass
