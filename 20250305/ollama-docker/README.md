In order to use ollama in docker I ran the following:

`docker run --rm -v ollama:/root/.ollama -p 11434:11434 --name ollama2 ollama/ollama`

It created a docker container named ollama2, listening to port 11434

In order to use the ollama client console, you can run:

`docker exec -it ollama2 ollama run llama3.2:1b `

This will connect into the same container, but running the ollama console, where you can interact with the LLM.


