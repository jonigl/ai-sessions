from ollama import generate

response = generate('llama3.2:1b', 'Why is the sky blue?')
print(response['response'])
