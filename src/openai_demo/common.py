"""Common definitions for the OpenAI demo."""

import os
import secrets
import string
import tempfile

import openai
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

OPENAI_URL = os.getenv("OPENAI_URL", "")
OPENAI_KEY = os.getenv("OPENAI_KEY", "")
assert OPENAI_URL, "OPENAI_URL environment variable must be set"
assert OPENAI_KEY, "OPENAI_KEY environment variable must be set"
OPENAI_CLIENT = openai.OpenAI(base_url=OPENAI_URL, api_key=OPENAI_KEY)

LLM_MODEL = os.getenv("OPENAI_MODEL", "")
assert LLM_MODEL, "OPENAI_MODEL environment variable must be set"

console = Console()

# Generate a random temporary directory for this run of the demo.
# Do not create the directory yet, as it will be created on demand when the first file is written.
rand_str = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(12))
TEMP_DIR = f"{tempfile.gettempdir()}/openai_demo/{rand_str}"
console.print(f"Using temporary directory: {TEMP_DIR}")
