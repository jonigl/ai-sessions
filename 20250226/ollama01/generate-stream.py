from ollama import generate

for part in generate('llama3.2:1b', 'Why is the sky blue?', stream=True):
  print(part['response'], end='', flush=True)
