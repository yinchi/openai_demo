import os

import openai
from dotenv import load_dotenv
from openai.types.responses import Response, ResponseInputParam
from rich.console import Console
from rich.markdown import Markdown
from rich.style import Style

load_dotenv()

OPENAI_URL = os.getenv("OPENAI_URL")
OPENAI_KEY = os.getenv("OPENAI_KEY")
CLIENT = openai.OpenAI(base_url=OPENAI_URL, api_key=OPENAI_KEY)
MODEL = os.getenv("OPENAI_MODEL")

console = Console()


def _handle_prompt(
    *,
    model: str | None,
    input_text: str | ResponseInputParam,
    previous_response_id: str | None = None,
) -> str | None:
    """Prompt the OpenAI API and display the response.

    Args:
        model: The model to use for the response.
        input_text: The input text or messages for the model.
        previous_response_id: The ID of the previous response, if any.

    Returns:
        The ID of the new response, or None if no response was generated.
    """

    response_id: str | None = None

    with console.status(
        "[bright_black]Thinking...",
        spinner="dots",
        spinner_style=Style(color="bright_black"),
    ):
        response = CLIENT.responses.create(
            model=model,
            input=input_text,
            previous_response_id=previous_response_id,
        )
        assert isinstance(response, Response)
        response_id = response.id
        console.print(Markdown(response.output_text))

    return response_id


def main() -> None:
    """Entry point for the demo script."""

    response_id: str | None = None

    while True:
        try:
            user_input = console.input("[bold green]Prompt: [/bold green]")
        except EOFError:
            # Handle Ctrl+D / EOF to exit the program gracefully
            print()
            return

        if not user_input.strip():
            # Empty prompt, prompt again without calling the API
            continue

        response_id = _handle_prompt(
            model=MODEL,
            previous_response_id=response_id,
            input_text=[{"role": "user", "content": user_input}],
        )
        print()  # Blank line between response and next prompt
