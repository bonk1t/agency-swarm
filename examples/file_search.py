#!/usr/bin/env python3
"""
Simple FileSearch Example - Agency Swarm v1.x

This example demonstrates how to use the FileSearch tool with citations.
The agent indexes PDF files for search; FileSearch must be added explicitly.
"""

import asyncio
import os
import sys
from pathlib import Path

# Path setup for standalone examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agency_swarm import Agency, Agent
from agency_swarm.utils.citation_extractor import display_citations, extract_vector_store_citations


async def main():
    """Demonstrate FileSearch functionality with citations."""

    print("🚀 Simple FileSearch Example")
    print("=" * 30)

    # Use the data directory with research files
    examples_dir = Path(__file__).parent
    docs_dir = examples_dir / "data"

    if not docs_dir.exists() or not list(docs_dir.glob("*.pdf")):
        print(f"❌ Error: No .pdf files found in: {docs_dir}")
        print("   Please ensure there are PDF files in the data directory.")
        return

    pdf_files = list(docs_dir.glob("*.pdf"))
    print(f"📁 Found {len(pdf_files)} PDF file(s) in: {docs_dir}")

    # Create an agent that can search files with citations
    search_agent = Agent(
        name="SearchAgent",
        instructions="You are a document search assistant. Use your FileSearch tool to find information and provide clear answers with citations.",
        files_folder=str(docs_dir),
        include_search_results=True,  # Enable citation extraction
    )
    # FileSearch is no longer added automatically; add it explicitly
    if search_agent._associated_vector_store_id:
        search_agent.file_manager.add_file_search_tool(search_agent._associated_vector_store_id)

    # Create agency
    agency = Agency(
        search_agent,
        shared_instructions="Demonstrate FileSearch with citations.",
    )

    # Wait for file processing
    print("⏳ Processing files...")
    await asyncio.sleep(3)

    # Test search with a specific question
    question = "What is the secret phrase in the PDF file?"
    print(f"\n❓ Question: {question}")

    try:
        response = await agency.get_response(question)
        print(f"🤖 Answer: {response.final_output}")

        # Extract and display citations using the utility function
        citations = extract_vector_store_citations(response)
        display_citations(citations, "vector store")

        # Check if we got the expected answer
        if "FIRST PDF SECRET PHRASE" in response.final_output.upper():
            print("✅ Correct answer found!")
        else:
            print("ℹ️  Try different questions from the research data")

    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n🎯 Usage Tips:")
    print("   • Add more .pdf files to the data/ directory")
    print("   • Citations show which files contain the answers")
    print("   • Vector store persists between runs")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
