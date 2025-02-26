from ollama import ChatResponse, chat
import urllib.request


def weather_forecast(location: str) -> str:
  """
  Get the weather forecast for a location

  Args:
    location (str): The location to get the weather forecast for

  Returns:
    str: The weather forecast
  """
  contents = urllib.request.urlopen(f'https://wttr.in/{location}?format=%C+%t').read()
  return contents.decode('utf-8')
  
  

def add_two_numbers(a: int, b: int) -> int:
  """
  Add two numbers

  Args:
    a (int): The first number
    b (int): The second number

  Returns:
    int: The sum of the two numbers
  """

  # The cast is necessary as returned tool call arguments don't always conform exactly to schema
  # E.g. this would prevent "what is 30 + 12" to produce '3012' instead of 42
  return int(a) + int(b)


def subtract_two_numbers(a: int, b: int) -> int:
  """
  Subtract two numbers
  """

  # The cast is necessary as returned tool call arguments don't always conform exactly to schema
  return int(a) - int(b)


# Tools can still be manually defined and passed into chat
subtract_two_numbers_tool = {
  'type': 'function',
  'function': {
    'name': 'subtract_two_numbers',
    'description': 'Subtract two numbers',
    'parameters': {
      'type': 'object',
      'required': ['a', 'b'],
      'properties': {
        'a': {'type': 'integer', 'description': 'The first number'},
        'b': {'type': 'integer', 'description': 'The second number'},
      },
    },
  },
}

messages = [{'role': 'user', 'content': 'what is the weather in Madrid?'}]
print('Prompt:', messages[0]['content'])

available_functions = {
  'add_two_numbers': add_two_numbers,
  'subtract_two_numbers': subtract_two_numbers,
  'weather_forecast': weather_forecast,
}

response: ChatResponse = chat(
  'llama3.2:1b',
  messages=messages,
  tools=[add_two_numbers, subtract_two_numbers_tool, weather_forecast],
)

if response.message.tool_calls:
  # There may be multiple tool calls in the response
  for tool in response.message.tool_calls:
    # Ensure the function is available, and then call it
    if function_to_call := available_functions.get(tool.function.name):
      print('Calling function:', tool.function.name)
      print('Arguments:', tool.function.arguments)
      output = function_to_call(**tool.function.arguments)
      print('Function output:', output)
    else:
      print('Function', tool.function.name, 'not found')

# Only needed to chat with the model using the tool call results
if response.message.tool_calls:
  # Add the function response to messages for the model to use
  messages.append(response.message)
  messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})

  # Get final response from model with function outputs
  final_response = chat('llama3.2:1b', messages=messages)
  print('Final response:', final_response.message.content)

else:
  print('No tool calls returned from model')
