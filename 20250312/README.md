# 20250312

We continue learning about the basics of RAG (Retrieval-Augmented Generation) and how it works. We also learn how to set up a CLI tool to use RAG with Ollama.

## ollama-rag-cli

Check out the [RAG CLI](20250312/ollama-rag-cli/README.md) tool that allows you to chat with your personal documents using local language models through Ollama.
Check in general [langchain-community API reference](https://python.langchain.com/api_reference) but in particular:
- Ollama integration, check out the [Ollama API reference](https://python.langchain.com/api_reference/ollama/index.html)
- ChromeDB integration, check out the [ChromaDB API reference](https://python.langchain.com/api_reference/chroma/index.html)
- Document loaders, check out the [Document API reference](https://python.langchain.com/api_reference/community/document_loaders.html)

### Using it for Mulesoft Docs

I have created some forks of the Mulesoft Docs [Hosting](https://github.com/jonigl/mulesoft-docs-hosting-test) and [Runtime](https://github.com/jonigl/mulesoft-docs-mule-runtime-test) repositories to test the RAG CLI tool. The repository contains a few documents that can be used to test the RAG CLI tool. 

You can clone the repositories and use the RAG CLI tool to chat with the documents.
For example to embed the documents from the Mulesoft Docs Runtime repository, you can run the following command:

```bash
python3 orag.py embed mulesoft-docs-mule-runtime-test/output_text_directory
```
and then chat with the documents using the following command:

```bash
python3 orag.py chat
```
Tip: You can also use the `--help` option to see the available commands and options.
