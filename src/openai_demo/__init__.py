"""OpenAI API demo script with tool calls.

Executes a simple prompt loop with tools and Markdown rendering in the terminal using Rich.
"""

import json

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageCustomToolCallParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionMessageToolCallUnionParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_custom_tool_call_param import Custom as CustomParam
from openai.types.chat.chat_completion_message_tool_call_param import Function as FunctionParam
from rich.markdown import Markdown
from rich.style import Style

from .common import LLM_MODEL, OPENAI_CLIENT, console
from .tools import TOOL_HANDLERS, TOOLS

SYSTEM_PROMPT = (
    "You are a helpful assistant. You are running in a console application with "
    "Markdown support via the `rich` Python library."
)


def _execute_tool_call(tool_call: ChatCompletionMessageFunctionToolCall) -> str:
    """Execute a tool call requested by the model."""
    function_name = tool_call.function.name
    function_args = tool_call.function.arguments or "{}"

    try:
        parsed_args = json.loads(function_args)
    except json.JSONDecodeError as exc:
        return f"Tool arguments were not valid JSON: {exc}"

    tool = TOOL_HANDLERS.get(function_name)
    if tool is None:
        return f"Unknown tool: {function_name}"

    try:
        result = tool(**parsed_args)
    except TypeError as exc:
        return f"Tool arguments were invalid for {function_name}: {exc}"
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return f"Tool {function_name} failed: {exc}"

    return str(result)


def tool_call_to_param(
    tool_call: ChatCompletionMessageToolCallUnion,
) -> ChatCompletionMessageToolCallUnionParam:
    """Convert a ChatCompletionMessageFunctionToolCall to a ChatCompletionFunctionToolParam."""
    if isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
        return ChatCompletionMessageFunctionToolCallParam(
            type=tool_call.type,
            id=tool_call.id,
            function=FunctionParam(
                name=tool_call.function.name,
                arguments=tool_call.function.arguments,
            ),
        )
    return ChatCompletionMessageCustomToolCallParam(
        type=tool_call.type,
        id=tool_call.id,
        custom=CustomParam(
            name=tool_call.custom.name,
            input=tool_call.custom.input,
        ),
    )


def _handle_prompt(
    *,
    messages: list[ChatCompletionMessageParam],
) -> None:
    """Prompt the OpenAI-compatible API, execute tools, and display the reply.

    Args:
        model: The model to use for the response.
        messages: The full conversation history in chat-completions format.
    """
    with console.status(
        "[bright_black]Thinking...",
        spinner="dots",
        spinner_style=Style(color="bright_black"),
    ):
        response = OPENAI_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
        )

    # For simplicity, always take the first choice.
    response_message = response.choices[0].message

    # While the response contains tool calls, execute them, append the results to the conversation,
    # and re-query for a new response.
    while response_message.tool_calls:
        # Append the assistant message with the tool calls to the conversation history.
        messages.append(
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=response_message.content,
                tool_calls=list(map(tool_call_to_param, response_message.tool_calls)),
            )
        )

        # Extract the list of tool calls, checking that they are all of type "function"
        # (the only type currently supported).
        tool_calls: list[ChatCompletionMessageFunctionToolCall] = []

        for tool_call in response_message.tool_calls:
            if tool_call.type == "function":
                tool_calls.append(tool_call)
            else:
                console.print(
                    Markdown(f"**Unsupported tool call type:** {tool_call.type}"),
                    style="red",
                )

                # Abort the prompt handler.
                break

        # Execute each tool call and append the results to the conversation history
        for tool_call in tool_calls:
            tool_result = _execute_tool_call(tool_call)
            messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_call.id,
                    content=tool_result,
                )
            )

        with console.status(
            "[bright_black]Thinking...",
            spinner="dots",
            spinner_style=Style(color="bright_black"),
        ):
            response = OPENAI_CLIENT.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS,
            )

        response_message = response.choices[0].message

    # Once there are no more tool calls, display the final response content.
    if response_message.content:
        messages.append(
            ChatCompletionAssistantMessageParam(role="assistant", content=response_message.content)
        )
        console.print(Markdown(response_message.content))
    elif response_message.refusal:
        messages.append(
            ChatCompletionAssistantMessageParam(role="assistant", refusal=response_message.refusal)
        )
        console.print(
            Markdown(f"**Model refused to answer:** {response_message.refusal}"),
            style="red",
        )
    else:
        # No content or refusal - just print a blank line and move on.
        console.print()


def main() -> None:
    """Entry point for the demo script."""
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=SYSTEM_PROMPT),
    ]

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

        messages.append(ChatCompletionUserMessageParam(role="user", content=user_input))
        _handle_prompt(messages=messages)
        print()  # Blank line between response and next prompt
