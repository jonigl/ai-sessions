# mcp-client

How to build a LLM-powered chatbot client that connects to MCP servers. Check this [tutorial](https://modelcontextprotocol.io/quickstart/client#python)

## Prerequisites
- [uv](https://github.com/astral-sh/uv)
- Python 10
- Anthropic key

## Set environment variable

Copy the `.env.example` file to `.env` and set your Anthropic key:

```bash
cp .env.example .env
```

Then open `.env` and set the `ANTHROPIC_API_KEY` variable to your Anthropic key.

```bash
ANTHROPIC_API_KEY=your_anthropic_key
```


## Setup virtual environment

```bash
uv venv
source .venv/bin/activate
``` 

## Run the client

```bash
uv run client.py ../server/weather/weather.py
```

This will install dependencies and start the server. You will see a chat interface in your terminal.



