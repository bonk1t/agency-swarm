import base64
import os
import shutil
import uuid
from pathlib import Path

import pytest
from agents import ModelSettings, ToolCallItem
from openai import AsyncOpenAI

from agency_swarm import Agency, Agent

# Ensure API key is available for tests that make real calls
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
IS_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"

# Conditional skip for tests requiring OpenAI API key if not in GITHUB_ACTIONS
# In GitHub Actions, these tests are expected to run with secrets.
requires_openai_api = pytest.mark.skipif(not OPENAI_API_KEY and not IS_GITHUB_ACTIONS, reason="Requires OPENAI_API_KEY")


@pytest.fixture(scope="module")
async def real_openai_client():
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set, skipping tests that make real API calls.")
    return AsyncOpenAI(api_key=OPENAI_API_KEY)

@requires_openai_api
@pytest.mark.asyncio
async def test_agent_processes_message_files_attachment(real_openai_client: AsyncOpenAI, tmp_path: Path):
    """
    Tests that an agent can receive a file ID via `file_ids` parameter,
    and OpenAI automatically makes the file content available to the LLM.
    No custom tools are needed - OpenAI handles file processing automatically.
    Uses a REAL OpenAI client for file upload and REAL agent execution.
    """
    # Use existing rich test PDF from v0.X tests
    test_pdf_path = Path("tests/data/files/test-pdf.pdf")
    assert test_pdf_path.exists(), f"Test PDF not found at {test_pdf_path}"

    uploaded_real_file = await real_openai_client.files.create(file=test_pdf_path.open("rb"), purpose="assistants")
    attached_file_id = uploaded_real_file.id
    print(f"TEST: Uploaded file {test_pdf_path.name} for attachment, ID: {attached_file_id}")

    # 2. Initialize Agent WITHOUT any custom file processing tools
    # OpenAI will automatically process the attached file and make content available to the LLM
    attachment_tester_agent = Agent(
        name="AttachmentTesterAgentReal",
        instructions=(
            "You are a helpful assistant. When files are attached, you can read their content directly. "
            "Answer questions about the file content accurately."
        ),
        model_settings=ModelSettings(temperature=0.0),
    )
    attachment_tester_agent._openai_client = real_openai_client

    # 3. Setup a minimal real Agency and ThreadManager for agent.get_response()
    agency = Agency(attachment_tester_agent, user_context=None)
    thread_manager = attachment_tester_agent._thread_manager
    assert thread_manager is not None, "ThreadManager not set by Agency"

    # 4. Call get_response with file_ids - OpenAI will automatically process the file
    test_chat_id = f"test_msg_files_real_chat_{uuid.uuid4().hex[:8]}"
    message_to_agent = "What content do you see in the attached PDF file? Please summarize what you find."

    print(f"TEST: Calling get_response for agent '{attachment_tester_agent.name}' with file_ids: [{attached_file_id}]")
    response_result = await attachment_tester_agent.get_response(
        message_to_agent, chat_id=test_chat_id, file_ids=[attached_file_id]
    )

    assert response_result is not None
    assert response_result.final_output is not None
    print(f"TEST: Agent final output: {response_result.final_output}")

    # 5. Verify the agent could access the file content automatically
    # The LLM should be able to read the PDF content without any custom tools
    # The test PDF contains a secret phrase we can verify
    response_lower = response_result.final_output.lower()
    assert len(response_result.final_output) > 20, (
        f"Response too short, suggests file content was not processed. Response: {response_result.final_output}"
    )

    # Look for the secret phrase that should be in the PDF
    secret_phrase_found = "first pdf secret phrase" in response_lower
    assert secret_phrase_found, (
        f"Expected secret phrase 'FIRST PDF SECRET PHRASE' not found in response. "
        f"This suggests the PDF content was not made available to the LLM. "
        f"Response: {response_result.final_output}"
    )

    # 6. Verify NO custom tool calls were made (since OpenAI handles file processing automatically)
    tool_calls_found = False
    for item in response_result.new_items:
        if isinstance(item, ToolCallItem):
            tool_calls_found = True
            print(f"Unexpected tool call found: {item.raw_item}")

    # We expect NO tool calls since OpenAI processes files automatically
    assert not tool_calls_found, (
        "No tool calls should be found since OpenAI automatically processes file attachments. "
        "The presence of tool calls suggests the implementation is incorrectly trying to use custom tools."
    )

    # 7. Cleanup the test file
    try:
        await real_openai_client.files.delete(attached_file_id)
        print(f"Cleaned up file {attached_file_id}")
    except Exception as e:
        print(f"Warning: Failed to clean up file {attached_file_id}: {e}")


@requires_openai_api
@pytest.mark.asyncio
async def test_multi_file_type_processing(real_openai_client: AsyncOpenAI, tmp_path: Path):
    """
    Tests that an agent can process PDF files automatically via OpenAI's Responses API file processing.

    NOTE: The OpenAI Responses API with input_file type only supports PDF files for direct attachment.
    Other file types (TXT, CSV, images) are supported through different mechanisms:
    - Vector Stores/File Search (for RAG functionality)
    - Code Interpreter (for code execution with files)

    This test focuses on the direct file attachment capability which is PDF-only.
    Uses the existing rich test PDF from the v0.X test suite.
    """
    # Use the existing rich test PDF with secret phrase
    test_pdf_path = Path("tests/data/files/test-pdf.pdf")
    assert test_pdf_path.exists(), f"Test PDF not found at {test_pdf_path}"

    # Upload PDF file to OpenAI
    with open(test_pdf_path, "rb") as f:
        uploaded_file = await real_openai_client.files.create(file=f, purpose="assistants")
        file_id = uploaded_file.id
        print(f"Uploaded {test_pdf_path.name}, got ID: {file_id}")

    try:
        # Create an agent WITHOUT custom file processing tools
        # OpenAI will automatically process PDF files and make content available
        file_processor_agent = Agent(
            name="FileProcessorAgent",
            instructions="""You are an agent that can read and analyze PDF files automatically.
            When PDF files are attached, you can access their content directly.
            Extract and summarize key information from the PDF content accurately.""",
            model_settings=ModelSettings(temperature=0.0),
        )
        file_processor_agent._openai_client = real_openai_client

        # Initialize agency for the agent
        agency = Agency(file_processor_agent, user_context=None)

        # Test processing the PDF file
        question = "What secret phrase do you find in this PDF file?"
        expected_content = "FIRST PDF SECRET PHRASE"

        # Process the PDF file - OpenAI will automatically make file content available
        chat_id = f"test_file_processing_{uuid.uuid4().hex[:8]}"
        response_result = await file_processor_agent.get_response(question, chat_id=chat_id, file_ids=[file_id])

        # Verify response
        assert response_result is not None
        print(f"Response for {test_pdf_path.name}: {response_result.final_output}")

        # Use case-insensitive search for matching
        response_lower = response_result.final_output.lower()
        expected_lower = expected_content.lower()

        # With temperature=0, responses should be deterministic
        content_found = expected_lower in response_lower

        assert content_found, (
            f"Expected content '{expected_content}' not found in response for {test_pdf_path.name}. "
            f"This suggests OpenAI did not make the PDF file content available to the LLM. "
            f"Response: {response_result.final_output}"
        )

        # Verify NO custom tool calls were made (OpenAI processes PDFs automatically)
        tool_calls_found = False
        for item in response_result.new_items:
            if isinstance(item, ToolCallItem):
                tool_calls_found = True
                print(f"Unexpected tool call found for {test_pdf_path.name}: {item.raw_item}")

        assert not tool_calls_found, (
            f"No tool calls should be found for {test_pdf_path.name} since OpenAI automatically processes PDF file attachments. "
            f"The presence of tool calls suggests the implementation is incorrectly trying to use custom tools."
        )

    finally:
        # Cleanup: Delete uploaded file from OpenAI
        try:
            await real_openai_client.files.delete(file_id=file_id)
            print(f"Cleaned up file {file_id}")
        except Exception as e:
            print(f"Error cleaning up file {file_id}: {e}")

@requires_openai_api
@pytest.mark.asyncio
async def test_file_search_tool(real_openai_client: AsyncOpenAI, tmp_path: Path):
    """
    Tests that an agent can use FileSearch tool to process files.
    """
    # Use the test txt file
    test_pdf_path = Path("tests/data/files/favorite_books.txt")
    assert test_pdf_path.exists(), f"Test PDF not found at {test_pdf_path}"

    # Upload PDF file to OpenAI
    tmp_dir = Path("tests/data/files/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file_path = tmp_dir / "favorite_books.txt"
    shutil.copy(test_pdf_path, tmp_file_path)

    try:
        # Create an agent WITHOUT custom file processing tools
        # Library will automatically add FileSearch tool
        file_search_agent = Agent(
            name="FileSearchAgent",
            instructions="""You are an agent that can read and analyze text files.""",
            model_settings=ModelSettings(temperature=0.0),
            files_folder=tmp_dir,
        )
        file_search_agent._openai_client = real_openai_client

        parent = tmp_dir.parent
        base_name = tmp_dir.name
        # Look for folders like base_name_vs_*
        candidates = list(parent.glob(f"{base_name}_vs_*"))
        if candidates:
            # Use the first match
            folder_path = candidates[0]
        else:
            folder_path = ""

        assert folder_path != "", "No vector store folder found"

        # Initialize agency for the agent
        agency = Agency(file_search_agent, user_context=None)

        question = "What is the name of the 4th book in the list?"

        chat_id = f"test_file_search_{uuid.uuid4().hex[:8]}"
        response_result = await agency.get_response(question, recipient_agent=file_search_agent, chat_id=chat_id)

        # Verify response
        assert response_result is not None
        print(f"Response for {test_pdf_path.name}: {response_result.final_output}")

        assert "hobbit" in response_result.final_output.lower()

    finally:
        # Cleanup: Delete uploaded file from OpenAI and temp directory
        try:
            for file in folder_path.glob("*"):
                file_id = file_search_agent.get_id_from_file(file)
                if file_id:
                    await real_openai_client.files.delete(file_id=file_id)
                    print(f"Cleaned up file {file.name}")
                os.remove(file)
            vector_store_id = folder_path.name.split("_vs_")[-1]
            await real_openai_client.vector_stores.delete(vector_store_id=f"vs_{vector_store_id}")
            print(f"Cleaned up vector store {folder_path.name}")
            os.rmdir(folder_path)
            print(f"Cleaned up folder {folder_path.name}")
        except Exception as e:
            print(f"Error cleaning up: {e}, dir: {tmp_dir.glob("*")}")

@requires_openai_api
@pytest.mark.asyncio
async def test_code_interpreter_tool(real_openai_client: AsyncOpenAI, tmp_path: Path):
    """
    Tests that an agent can read and execute code using CodeInterpreter tool.
    """
    test_pdf_path = Path("tests/data/files/test-python.py")
    assert test_pdf_path.exists(), f"Test PDF not found at {test_pdf_path}"

    tmp_dir = Path("tests/data/files/tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file_path = tmp_dir / "test-python.py"
    shutil.copy(test_pdf_path, tmp_file_path)

    try:
        code_interpreter_agent = Agent(
            name="CodeInterpreterAgent",
            instructions="""You are an agent that can read and execute code using CodeInterpreter tool.""",
            model_settings=ModelSettings(temperature=0.0),
            files_folder=tmp_dir,
        )
        code_interpreter_agent._openai_client = real_openai_client

        parent = tmp_dir.parent
        base_name = tmp_dir.name
        # Look for folders like base_name_vs_*
        candidates = list(parent.glob(f"{base_name}_vs_*"))
        if candidates:
            # Use the first match
            folder_path = candidates[0]
        else:
            folder_path = ""

        assert folder_path != "", "No vector store folder found"

        # Initialize agency for the agent
        agency = Agency(code_interpreter_agent, user_context=None)

        # Test the simple usage of the code interpreter tool (answer is always 37)
        question = """
        Use CodeInterpreter tool to execute this script and tell me the results:
        ```import random\nrandom.seed(115)\nprint(random.randint(1, 100))```
        """

        chat_id = f"test_code_interpreter_{uuid.uuid4().hex[:8]}"
        response_result = await agency.get_response(question, recipient_agent=code_interpreter_agent, chat_id=chat_id)

        # Verify response
        assert response_result is not None
        assert "37" in response_result.final_output.lower()

        # Execute python script (answer is always 14910)
        query = "Run test-python script, return me its results and tell me exactly what you did to get them."
        response_result = await agency.get_response(query, recipient_agent=code_interpreter_agent, chat_id=chat_id)

        assert response_result is not None
        assert "14910" in response_result.final_output.lower()

    finally:
        # Cleanup: Delete uploaded file from OpenAI and temp directory
        try:
            for file in folder_path.glob("*"):
                file_id = code_interpreter_agent.get_id_from_file(file)
                if file_id:
                    await real_openai_client.files.delete(file_id=file_id)
                    print(f"Cleaned up file {file.name}")
                os.remove(file)
            vector_store_id = folder_path.name.split("_vs_")[-1]
            await real_openai_client.vector_stores.delete(vector_store_id=f"vs_{vector_store_id}")
            print(f"Cleaned up vector store {folder_path.name}")
            os.rmdir(folder_path)
            print(f"Cleaned up folder {folder_path.name}")
        except Exception as e:
            print(f"Error cleaning up: {e}, dir: {tmp_dir.glob("*")}")

@requires_openai_api
@pytest.mark.asyncio
async def test_agent_vision_capabilities(real_openai_client: AsyncOpenAI, tmp_path: Path):
    """
    Tests that an agent can process images using OpenAI's vision capabilities.
    Uses the input_image format with base64 encoded images.
    Uses the pre-generated example images since vision requires actual image files.
    """

    def image_to_base64(image_path: Path) -> str:
        """Convert image file to base64 string."""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string

    # Use the example images since they're actual image files (not text files)
    test_images = [
        (Path("examples/data/shapes_and_text.png"), "How many shapes do you see in this image?", "three"),
        (Path("examples/data/shapes_and_text.png"), "What text do you see in this image?", "VISION TEST 2024"),
    ]

    # Verify test images exist
    for image_path, _, _ in test_images:
        assert image_path.exists(), f"Test image not found at {image_path}"

    # Create a vision-capable agent with temperature=0 for deterministic responses
    vision_agent = Agent(
        name="VisionAgent",
        instructions="""You are an expert vision AI that can analyze images accurately.
        When images are provided, examine them carefully and answer questions about their content.
        Be precise and specific in your descriptions.""",
        model_settings=ModelSettings(temperature=0.0),
    )
    vision_agent._openai_client = real_openai_client

    # Initialize agency for the agent
    agency = Agency(vision_agent, user_context=None)

    # Test processing each image
    for image_path, question, expected_content in test_images:
        print(f"\nTesting vision processing of {image_path.name}")

        # Convert image to base64
        b64_image = image_to_base64(image_path)

        # Create message with input_image format
        message_with_image = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": f"data:image/png;base64,{b64_image}",
                    }
                ],
            },
            {"role": "user", "content": question},
        ]

        # Process the image - OpenAI will automatically handle vision processing
        chat_id = f"test_vision_{uuid.uuid4().hex[:8]}"
        response_result = await vision_agent.get_response(message_with_image, chat_id=chat_id)

        # Verify response
        assert response_result is not None
        assert response_result.final_output is not None
        print(f"Vision response for {image_path.name}: {response_result.final_output}")

        # Use case-insensitive search for matching
        response_lower = response_result.final_output.lower()
        expected_lower = expected_content.lower()

        # With temperature=0, responses should be deterministic
        content_found = expected_lower in response_lower

        assert content_found, (
            f"Expected content '{expected_content}' not found in vision response for {image_path.name}. "
            f"This suggests the vision processing failed or the model couldn't see the image content. "
            f"Response: {response_result.final_output}"
        )

        # Verify NO custom tool calls were made (OpenAI processes vision automatically)
        tool_calls_found = False
        for item in response_result.new_items:
            if isinstance(item, ToolCallItem):
                tool_calls_found = True
                print(f"Unexpected tool call found for {image_path.name}: {item.raw_item}")

        # We expect NO tool calls since OpenAI processes vision automatically
        assert not tool_calls_found, (
            f"No tool calls should be found for {image_path.name} since OpenAI automatically processes vision. "
            f"The presence of tool calls suggests the implementation is incorrectly trying to use custom tools."
        )
