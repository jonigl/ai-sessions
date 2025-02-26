from ollama import chat

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
    {
    'role': 'system',
    'content': 'You answer talking like a pirate. DON\'T USE MARKDOWN EVER.', # 'content': 'You are barbie.'
  },
]

for part in chat('llama3.2:1b', messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)

print()
