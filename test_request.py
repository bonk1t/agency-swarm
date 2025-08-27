import asyncio

import httpx


async def main():
    url = "http://localhost:8080/agency1/get_response"
    payload = {
        "message": "Write a 200 word poem",
        # "file_ids": ["file-PV4zcu73aVYeHxDvb4g4jW"],
        "file_urls": {
            # "sample_pdf.pdf": "https://manuals.plus/m/2c4c0fc110814f4ee7d025333b7fc0c2f15db5a4bb7bb2e00273d8cd6a54d510.pdf",
            # "sample_doc": "https://www.sample-videos.com/doc/Sample-doc-file-100kb.doc",
            # "sample_docx": "https://www.cte.iup.edu/cte/Resources/DOCX_TestPage.docx",
            # "sample_image": "http://localhost:7860/test-html.html",
        },
    }
    headers = {"Authorization": "Bearer 123", "x-agency-log-id": "123"}
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(url, json=payload, headers=headers)
        print("main() response:", response.json())


async def main2():
    url = "http://localhost:8080/agency1/get_response"
    payload = {
        "message": "Write a 200 word poem",
        # "file_ids": ["file-PV4zcu73aVYeHxDvb4g4jW"],
        "file_urls": {
            # "sample_pdf.pdf": "https://manuals.plus/m/2c4c0fc110814f4ee7d025333b7fc0c2f15db5a4bb7bb2e00273d8cd6a54d510.pdf",
            # "sample_doc": "https://www.sample-videos.com/doc/Sample-doc-file-100kb.doc",
            # "sample_docx": "https://www.cte.iup.edu/cte/Resources/DOCX_TestPage.docx",
            # "sample_image": "http://localhost:7860/test-html.html",
        },
    }
    headers = {"Authorization": "Bearer 123", "x-agency-log-id": "123"}
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(url, json=payload, headers=headers)
        print("main2() response:", response.json())


async def run_concurrent():
    """Run main and main2 concurrently"""
    await asyncio.gather(main(), main2())


if __name__ == "__main__":
    asyncio.run(run_concurrent())
    # asyncio.run(main_stream())
    # print("--------------------------------")
    # asyncio.run(main_logs())
    # asyncio.run(main_routes())
