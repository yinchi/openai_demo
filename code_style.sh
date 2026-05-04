#!/usr/bin/env bash
# Run isort and ruff on the src/ directory

# Define the working directory as the directory containing this script
WORKING_DIR="$(dirname "$(realpath "$BASH_SOURCE")")"

echo cd "$WORKING_DIR"
cd "$WORKING_DIR"

echo
echo "====="
echo "ISORT"
echo "====="
uv run isort --show-files ./src/  # isort config in pyproject.toml

echo
echo "================"
echo "RUFF CHECK --FIX"
echo "================"
uv run ruff check --config ./.ruff.toml --fix ./src/ # ruff config in .ruff.toml

echo
echo "==========="
echo "RUFF FORMAT"
echo "==========="
uv run ruff format --config ./.ruff.toml ./src/ # ruff config in .ruff.toml

echo
echo "====="
echo "MYPY"
echo "====="
uv run mypy ./src/  # mypy config in pyproject.toml
