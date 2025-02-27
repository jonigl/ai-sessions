# Ollama 01 example

## Requirements

Install ollama 

https://ollama.com/

https://github.com/ollama/ollama?tab=readme-ov-file#ollama

```
ollama pull <model>:<version>
ollama run
ollama list
ollama rm
```

Ensure you have ollama in your machine and available. Pull llama 3.2:1b model:
```
ollama pull llama3.2:1b
```

Python version: 3.11.10

## Run the python script

```
python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python3 generate.py

// or for streamed answer

python3 generate-stream.py
```

# System role

We have also tested system role in order to change the assistant behavior and we tried a simple promt engineering to remove markdwon. 


![alt text](img/chat-example.png)


# Chat with history and stream
We also tested a chat with history and we could check that the context was there.

![alt text](img/chat-example.png)

![alt text](img/chat-history-stream.gif)


# Tools 

We have added a call to a public weather API.

Why is thi tool being called, if it has not even been documented? Maybe context is right, but if you create a different request, it will anyway use the "non-documented" tool. 

i.e. 
- "how much does it cost to rent a room in New York?" will call the function and then respond not using the function's response.
- "is it nice to be a famous singer?" called the function based in Paris, to then respond again without using that value.


Then I tested switching order of functions to be used array, and got this result:

```
response: ChatResponse = chat(
  'llama3.2:1b',
  messages=messages,
  tools= [subtract_two_numbers_tool, weather_forecast,add_two_numbers],
)
```

```
Prompt: is it nice to be a famous singer?
Calling function: subtract_two_numbers
Arguments: {'a': '1', 'b': '2'}
Function output: -1
Final response: Being a famous singer can have its advantages and disadvantages. On the one hand, being in the spotlight can bring a sense of recognition and admiration from fans and others. Many people find it rewarding to share their music with others and connect with them through their songs.
...
...

```

So definetly if it does not know how to choose a tool, might use the last one.

![alt text](img/tools.png)
