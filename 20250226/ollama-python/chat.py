from ollama import chat

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

response = chat('llama3.2:1b', messages=messages)
print(response['message']['content'])
