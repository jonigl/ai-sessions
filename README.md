# AI Sessions Repository

A collection of examples, tutorials, and projects exploring various AI technologies with a focus on local AI using Ollama.

## Overview

This repository contains organized examples of working with AI models locally through Ollama in different programming languages and configurations. Each directory is organized by date (YYYYMMDD format) and contains self-contained examples that can be run independently.

## Prerequisites

- [Ollama](https://ollama.com/) - Install from https://github.com/ollama/ollama
- Depending on the example:
  - Python 3.8+ with pip
  - Node.js and npm/yarn
  - Docker
  - Various language-specific dependencies (check each example's README)

## Repository Structure

### Sessions by Date

| Date | Session | Description |
|------|---------|-------------|
| [20250226](./20250226/) | Ollama Python | Examples of using Ollama with Python including generation, streaming, chat history, and tools |
| [20250305](./20250305/) | Ollama JS & Docker | JavaScript examples with Ollama and Docker setup instructions |
| [20250312](./20250312/) | Ollama RAG CLI | A command-line tool for Retrieval-Augmented Generation with Ollama |
| [20250326](./20250326/) | Deployments API | Tools for calling APIs using Ollama's function calling features |

## Sessions Overview

### Ollama Python (Feb 26, 2025)
Examples of using Ollama with Python, including:
- Basic text generation
- Streaming responses
- Chat with history
- Function calling and tools integration
- System role customization

### Ollama JavaScript (Mar 5, 2025)
Examples of using Ollama with JavaScript/TypeScript:
- Structured outputs
- Multimodal models using llava:13b

### Ollama Docker (Mar 5, 2025)
Instructions for running Ollama in Docker:
- Container setup
- Volume management
- Interactive console usage

### Ollama RAG CLI (Mar 12, 2025)
A complete command-line application for Retrieval-Augmented Generation:
- Document embedding
- Vector database storage
- Interactive chat with document context
- Support for multiple file types

### Deployments API (Mar 26, 2025)
Working with tools and function calling to interact with external APIs:
- API integrations with Ollama models
- Testing different models for tool usage
- Using `uv` Python package manager
- Environment configuration with dotenv

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ai-sessions.git
   cd ai-sessions
   ```

2. Navigate to any example directory and follow the README instructions for that specific example:
   ```bash
   cd 20250312/ollama-rag-cli
   # Follow the instructions in the README.md
   ```

3. Make sure to download the required Ollama models as specified in each example:
   ```bash
   ollama pull llama3:8b
   ollama pull nomic-embed-text
   # etc.
   ```

## License

[MIT License](LICENSE)
