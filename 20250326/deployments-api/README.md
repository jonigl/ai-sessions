# deployments-api

This tool allows you to call a Deployments API to get the response.

## Usage

This time we were using `uv` python package manager, so you need to install it first. For more information, you can check the [uv repo](https://github.com/astral-sh/uv).

```bash
pip install uv
```

Then you can run the following command to get the response from the API.

```bash
uv venv
source .venv/bin/activate
uv pip install .
```

Then you can run the following command to get the response from the API.

```bash
python main.py
```

## Environment Variables

This project uses `dotenv` to manage environment variables. You can create a `.env` file in the root directory and add the following variables.

```bash
API_HOST=http://localhost:3000
OLLAMA_MODEL=llama3.2:1b
```

Otherwise, it will use the default values.
