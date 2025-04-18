# MCP Weather Server

This is a simple weather mcp server that provides weather information based on user input.

## Prerequisites
- [uv](https://github.com/astral-sh/uv)
- Python 10

## Setup virtual environment

```bash
uv venv
source .venv/bin/activate
```

## How to run the server
To run the weather server, navigate to the `mcp/server/weather` directory and execute the following command:

```bash
uv run weather.py
```

This will install dependencies and start the server. You won't see any output.
