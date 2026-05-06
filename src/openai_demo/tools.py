"""Defines tools that the assistant can call."""

import json
import os
import shlex
import subprocess
from collections.abc import Callable
from typing import ParamSpec

from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionToolUnionParam,
)
from openai.types.shared_params import FunctionDefinition
from rich import prompt
from rich.markdown import Markdown

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
Directories are listed with trailing slashes and with a 'd' in the permissions column.

Equivalent to `eza -alhgF --smart-group --group-directories-first --color=never {pathname}`.
""",
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

# Summary of the `bc` Command-Line Tool

`bc` is a command-line utility that provides an arbitrary-precision calculator language. It is designed for both interactive use and for executing scripts.

## Core Features

- **Arbitrary Precision:** Unlike many standard calculators, `bc` can handle numbers with an arbitrary number of digits in both the integer and fractional parts.
- **Variable Support:** Supports both simple named variables and arrays.
- **Base Conversion:** Allows users to specify input (`ibase`) and output (`obase`) bases, ranging from 2 to 36 (or higher in some implementations).
- **Control Structures:** Includes standard programming constructs such as `if-else`, `while` loops, and `for` loops.
- **Functions:** Supports user-defined functions, including support for local variables (`auto`) and recursion.

## Key Variables

| Variable | Description |
| :--- | :--- |
| `scale` | Defines the number of digits after the decimal point for division and multiplication operations. Default is 0. |
| `ibase` | Defines the base for input numbers. |
| `obase` | Defines the base for output numbers. |
| `last` | An extension that stores the value of the last printed number. |

## Common Options

- `-l` (`--mathlib`): Defines the standard math library and sets the default `scale` to 20.
- `-q` (`--quiet`): Suppresses the normal GNU `bc` welcome message.
- `-i` (`--interactive`): Forces interactive mode.
- `-v` (`--version`): Prints version information and exits.

## Math Library (with `-l` option)

When the math library is loaded, the following functions become available:

- `s(x)`: Sine of `x` (in radians).
- `c(x)`: Cosine of `x` (in radians).
- `a(x)`: Arctangent of `x` (returns radians).
- `l(x)`: Natural logarithm of `x`.
- `e(x)`: Exponential function ($e^x$).
- `j(n, x)`: Bessel function of integer order `n` of `x`.

## Example Usage

**Basic Arithmetic:**
```bash
echo "10 + 5" | bc
# Output: 15
```

**Using Scale for Division:**
```bash
echo "scale=2; 10 / 3" | bc
# Output: 3.33
```

**Calculating Pi (using math library):**
```bash
echo "scale=10; 4*a(1)" | bc -l
# Output: 3.1415926535
```
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


@register_tool(
    ChatCompletionFunctionToolParam(
        type="function",
        function=FunctionDefinition(
            name="python_tool",
            description="""\
Execute the given Python code and return the output.

Prompts the user for confirmation before executing the code to prevent accidental execution of
harmful code. The code is automatically wrapped in a Markdown code block when printed to the
console for confirmation.

The code is executed using `subprocess.run` with a call to the Python interpreter,
and the output is captured and returned as a JSON object with `stdout`, `stderr`, and `returncode`
fields.  If execution is refused by the user, returns a JSON object with a `refused` field.
""",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute.",
                    }
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            strict=True,
        ),
    )
)
def python_tool(code: str) -> str:
    """Execute the given Python code and return the output."""
    console.print(f"[bright_black]Executing python_tool...[/bright_black]")
    console.print("[red bold]Execute the following Python code?[/red bold]")
    console.print(Markdown(f"```py\n{code}\n```"))
    confirm = prompt.Confirm.ask("[bold red]Execute?[/bold red]", default=False)
    if not confirm:
        return json.dumps({"refused": "Execution cancelled by user."})
    try:
        subprocess_result = subprocess.run(
            ["/usr/bin/python3", "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        return json.dumps(
            {
                "stdout": subprocess_result.stdout,
                "stderr": subprocess_result.stderr,
                "returncode": subprocess_result.returncode,
            }
        )
    except Exception as e:
        return json.dumps({"error": f"Error executing code: {e}"})

@register_tool(
    ChatCompletionFunctionToolParam(
        type="function",
        function=FunctionDefinition(
            name="python_file_tool",
            description="""\
Execute a Python file at the given path and return the output.
Prompts the user for confirmation before executing the code to prevent accidental execution of
harmful code. The content of the file is automatically wrapped in a Markdown code block when printed
to the console for confirmation.

The code is executed using `subprocess.run` with a call to the Python interpreter, and the output is
captured and returned as a JSON object with `stdout`, `stderr`, and `returncode` fields.  If
execution is refused by the user, returns a JSON object with a `refused` field.

The path must be within the current working directory or the temporary directory. If the tool
fails any of the checks on the path (does not exist, is not a file, is not readable, or is outside
of the allowed directories), it returns a JSON object with an `error` field describing the error.
""",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path of the Python file to execute.",
                    },
                    "tool_args": {
                        "type": "string",
                        "description": """\
Additional arguments to pass to the Python interpreter when executing the file. Should be a string
of command-line arguments, for example "arg1 arg2 --option=value --flag". These will be parsed
using `shlex.split` before being passed to the interpreter using `subprocess.run`.""",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            strict=True,
        ),
    )
)
def python_file_tool(path: str, tool_args: str) -> str:
    """Execute a Python file at the given path and return the output.
    
    Args:
        path: The path of the Python file to execute. Must be within the current working directory
            or the temporary directory.
        tool_args: Additional arguments to pass to the Python interpreter when executing the file.
    """

    console.print(f"[bright_black]Executing python_file_tool(path={path}, tool_args=\"{tool_args}\")...[/bright_black]")
    resolved_path = os.path.abspath(path)

    # Check that the path, when resolved, is within the current working directory or the temporary
    # directory.
    cwd = os.getcwd()
    temp_dir = os.path.abspath(TEMP_DIR)
    if not resolved_path.startswith(cwd) and not resolved_path.startswith(temp_dir):
        return json.dumps({
            "error": "Path must be within the current working directory or the temporary directory."
        })

    if not os.path.exists(resolved_path):
        return json.dumps({"error": "File does not exist."})

    if not os.path.isfile(resolved_path):
        return json.dumps({"error": "Path must be a file."})

    if not os.access(resolved_path, os.R_OK):
        return json.dumps({"error": "File is not readable."})
    
    if not prompt.Confirm.ask(f"[bold red]Execute Python file at {resolved_path}?[/bold red]", default=False):
        return json.dumps({"refused": "Execution cancelled by user."})

    try:
        subprocess_result = subprocess.run(
            # Use shlex.split to properly handle tool_args with spaces and special characters
            ["/usr/bin/python3", resolved_path] + shlex.split(tool_args),
            capture_output=True,
            text=True,
            check=False,
        )
        return json.dumps(
            {
                "stdout": subprocess_result.stdout,
                "stderr": subprocess_result.stderr,
                "returncode": subprocess_result.returncode,
            }
        )
    except Exception as e:
        return json.dumps({"error": f"Error executing code: {e}"})