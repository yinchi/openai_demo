"""Defines tools that the assistant can call."""

import os
import subprocess
from collections.abc import Callable

from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionToolUnionParam,
)
from openai.types.shared_params import FunctionDefinition

from .common import console


def eza_tool(p: str) -> str:
    """List files in the given directory."""
    console.print("[bright_black]Executing eza_tool...[/bright_black]")

    # Check that the path, when resolved, is within the current working directory
    # to prevent arbitrary file access.
    resolved_path = os.path.abspath(p)
    cwd = os.getcwd()
    if not resolved_path.startswith(cwd):
        return "Error: Path must be within the current working directory."

    sp = subprocess.run(
        f"eza -alhgF --smart-group --group-directories-first --color=never {p}",
        shell=True,
        capture_output=True,
        text=True,
        check=False,  # Don't raise an exception on non-zero exit codes - we'll handle it ourselves
    )
    return sp.stdout if sp.returncode == 0 else f"Error executing eza_tool: {sp.stderr}"


TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "eza_tool": eza_tool,
}

TOOLS: list[ChatCompletionToolUnionParam] = [
    ChatCompletionFunctionToolParam(
        type="function",
        function=FunctionDefinition(
            name="eza_tool",
            description="""\
List files and directories in the given directory.

Uses the `eza` command-line tool. Prints one entry per line, with a header row at the top.""",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to list files for.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            strict=True,
        ),
    )
]
