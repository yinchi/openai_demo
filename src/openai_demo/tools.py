"""Defines tools that the assistant can call."""

import os
import subprocess
from collections.abc import Callable
from typing import ParamSpec

from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionToolUnionParam,
)
from openai.types.shared_params import FunctionDefinition

from .common import TEMP_DIR, console

P = ParamSpec("P")

TOOL_HANDLERS: dict[str, Callable[..., str]] = {}
TOOLS: list[ChatCompletionToolUnionParam] = []


def register_tool(
    tool: ChatCompletionFunctionToolParam,
) -> Callable[[Callable[P, str]], Callable[P, str]]:
    """Register a callable and its chat-completions tool definition."""

    def decorator(func: Callable[P, str]) -> Callable[P, str]:
        tool_name = tool["function"]["name"]
        TOOL_HANDLERS[tool_name] = func
        TOOLS.append(tool)
        return func

    return decorator


@register_tool(
    ChatCompletionFunctionToolParam(
        type="function",
        function=FunctionDefinition(
            name="eza_tool",
            description="""\
List files and directories in the given directory.

Uses the `eza` command-line tool. Prints one entry per line, with a header row at the top.
Directories are listed with trailing slashes and with a 'd' in the permissions column.""",
            parameters={
                "type": "object",
                "properties": {
                    "pathname": {
                        "type": "string",
                        "description": "The path to list files for.",
                    }
                },
                "required": ["pathname"],
                "additionalProperties": False,
            },
            strict=True,
        ),
    )
)
def eza_tool(pathname: str) -> str:
    """List files in the given directory."""
    console.print(f"[bright_black]Executing eza_tool(pathname={pathname})...[/bright_black]")

    # Check that the path, when resolved, is within the current working directory
    # to prevent arbitrary file access.
    resolved_path = os.path.abspath(pathname)
    temp_dir = os.path.abspath(TEMP_DIR)
    cwd = os.getcwd()
    if not resolved_path.startswith(cwd) and not resolved_path.startswith(temp_dir):
        return (
            "Error: Path must be within the current working directory or the temporary directory."
        )

    sp = subprocess.run(
        f"eza -alhgF --smart-group --group-directories-first --color=never {pathname}",
        shell=True,
        capture_output=True,
        text=True,
        check=False,  # Don't raise an exception on non-zero exit codes - we'll handle it ourselves
    )
    return sp.stdout if sp.returncode == 0 else f"Error executing eza_tool: {sp.stderr}"


@register_tool(
    ChatCompletionFunctionToolParam(
        type="function",
        function=FunctionDefinition(
            name="bc_tool",
            description="""\
Evaluate a mathematical expression using the `bc` command-line tool.

Executes `echo '{expression}' | bc -l` and returns the output.
""",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            strict=True,
        ),
    )
)
def bc_tool(expression: str) -> str:
    """Evaluate a mathematical expression using the `bc` command-line tool."""
    console.print(f"[bright_black]Executing bc_tool(expression={expression})...[/bright_black]")
    sp = subprocess.run(
        f"echo '{expression}' | bc -l",
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    return sp.stdout.strip() if sp.returncode == 0 else f"Error executing bc_tool: {sp.stderr}"


@register_tool(
    ChatCompletionFunctionToolParam(
        type="function",
        function=FunctionDefinition(
            name="get_temp_dir",
            description="""\
Get the path to the temporary directory for this chat session.

Note that the temporary directory is not created until the first file is written to it, so the path
may not exist yet. The path will be different on each run of the demo to prevent conflicts between
runs.  Other tools can list, read, and write files within both the current
working directory and the temporary directory, but not outside of those directories, to prevent
arbitrary file access.
""",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            strict=True,
        ),
    )
)
def get_temp_dir() -> str:
    """Get the path to the temporary directory for this run of the demo."""
    console.print("[bright_black]Executing get_temp_dir()...[/bright_black]")
    return TEMP_DIR


@register_tool(
    ChatCompletionFunctionToolParam(
        type="function",
        function=FunctionDefinition(
            name="read_file",
            description="""\
Read content from a file at the given path.

The path must be within the current working directory or the temporary directory.
Returns an error if the path does not exist, is not a file, or is not readable.

When printing a file to the console, always wrap the content in a Markdown code block to preserve
formatting, for example:
```text
Green leaves on the tree
Sun shines bright upon the ground
Nature is so calm
```

Use an appropriate language identifier for the code block if the file contains code, for example
`py` for Python code, to enable syntax highlighting in the console.
""",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path of the file to read.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            strict=True,
        ),
    )
)
def read_file(path: str) -> str:
    """Read content from a file at the given path."""
    console.print(f"[bright_black]Executing read_file(path={path})...[/bright_black]")
    resolved_path = os.path.abspath(path)

    cwd = os.getcwd()
    temp_dir = os.path.abspath(TEMP_DIR)
    if not resolved_path.startswith(cwd) and not resolved_path.startswith(temp_dir):
        return (
            "Error: Path must be within the current working directory or the temporary directory."
        )

    if not os.path.exists(resolved_path):
        return "Error: File does not exist."

    if not os.path.isfile(resolved_path):
        return "Error: Path must be a file."

    if not os.access(resolved_path, os.R_OK):
        return "Error: File is not readable."

    try:
        with open(resolved_path) as f:
            return f.read()
    except PermissionError:
        return "Error: File is not readable."
    except OSError as e:
        return f"Error reading file: {e}"


@register_tool(
    ChatCompletionFunctionToolParam(
        type="function",
        function=FunctionDefinition(
            name="write_file",
            description="""\
Write content to a file at the given path.

The path must be within the current working directory or the designated temporary directory.
The file will only be written if it does not already exist to prevent overwriting existing
files.""",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to write the file to.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            strict=True,
        ),
    )
)
def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path."""
    if len(content) > 20:
        display_content = content[:20] + f"...({len(content)} characters)"
    else:
        display_content = content
    console.print(
        f"[bright_black]Executing write_file(path={path}, "
        f"content={display_content})...[/bright_black]"
    )
    resolved_path = os.path.abspath(path)

    # Check that the path, when resolved, is within the current working directory or the temporary
    # directory.
    cwd = os.getcwd()
    temp_dir = os.path.abspath(TEMP_DIR)
    if not resolved_path.startswith(cwd) and not resolved_path.startswith(temp_dir):
        return (
            "Error: Path must be within the current working directory or the temporary directory."
        )

    # Check that the resolved path is not a directory
    if os.path.isdir(resolved_path):
        return "Error: Path must not be a directory."

    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)

    # Fail if the file already exists to prevent overwriting existing files
    if os.path.exists(resolved_path):
        return "Error: File already exists. Will not overwrite existing files."

    try:
        with open(resolved_path, "w") as f:
            f.write(content)
        return f"File written successfully to {resolved_path}"
    except Exception as e:
        return f"Error writing file: {e}"
