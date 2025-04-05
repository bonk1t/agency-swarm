import mimetypes

# Register the MIME type for .xlsx files
mimetypes.add_type(
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"
)
mimetypes.add_type(
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"
)
mimetypes.add_type(
    "application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"
)

image_types = ["image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif"]

code_interpreter_types = [
    "application/csv",
    "image/jpeg",
    "image/gif",
    "image/png",
    "application/x-tar",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/xml",
    "text/xml",
    "application/zip",
    "text/csv",
]

dual_types = [
    "text/x-c",
    "text/x-csharp",
    "text/x-c++",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/html",
    "text/x-java",
    "application/json",
    "text/markdown",
    "application/pdf",
    "text/x-php",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/x-python",
    "text/x-script.python",
    "text/x-ruby",
    "text/x-tex",
    "text/plain",
    "text/css",
    "text/javascript",
    "application/x-sh",
    "application/typescript",
]


def get_file_purpose(file_path: str) -> str:
    """Get the appropriate file purpose for OpenAI API based on file type.

    Args:
        file_path: Path to the file

    Returns:
        Purpose string for OpenAI API
    """
    # Get file extension
    ext = file_path.lower().split(".")[-1]

    # Image files for vision
    if ext in ["jpg", "jpeg", "png", "gif", "webp"]:
        return "vision"

    # Files for search
    if ext in ["txt", "json", "csv", "md", "pdf", "py", "js", "html", "css"]:
        return "file_search"

    # Default to file_search for unknown types
    return "file_search"


def get_tools(file_path):
    """Returns the tools for the given file path"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        raise ValueError(f"Could not determine type for file: {file_path}")
    elif mime_type in dual_types:
        return [{"type": "file_search"}]
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")
