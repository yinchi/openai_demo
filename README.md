# OpenAI chat program demo

Command-line chat program that demonstrates how to use the OpenAI API for chat completions,
with:

- tool support
- conversation history (with /reset command to reset history)
- Markdown rendering

![screenshot of the demo](image.png)

## Setup

### Set up the LM Studio API server

1. Download and install the LM Studio API server:
   ```bash
   curl -fsSL https://lmstudio.ai/install.sh | bash
   ```
2. Download and launch a model:
   ```bash
   lms get google/gemma-4-26b-a4b
   lms load --gpu max -c 256000 --ttl 360000 google/gemma-4-26b-a4b
   ```
3. Start the API server:
   ```bash
   lms server start
   ```

### Set up `.env`

```bash
cp .env.example .env
```

Then edit the `.env` file to set the correct values for `OPENAI_URL`, `OPENAI_KEY`, and `OPENAI_MODEL`.
For local models such as LM Studio, the `OPENAI_KEY` can be set to any dummy value since authentication is not required.

### Run the demo

```bash
uv run demo
```
