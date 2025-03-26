from ollama import ChatResponse, chat
import urllib.request
import urllib.parse
import json
import dotenv
import os

dotenv.load_dotenv()

API_HOST = os.getenv('API_HOST') or 'localhost:3000'
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL') or 'PetrosStav/gemma3-tools:4b'

def get_deployments_from_api(fromDate: str = None, toDate: str = None) -> str:
    """
    Get deployments

    Args:
        fromDate (str): The start date of the deployments must be in the format ISO 8601 format. Optional.
        toDate (str): The end date of the deployments must be in the format ISO 8601 format. Optional.

    Returns:
        str: deployments in the system in the given date range in JSON format
    """

    # check if fromDate and toDate are provided
    params = ''
    if fromDate and toDate:
        params = f'?fromDate={fromDate}&toDate={toDate}'
    elif fromDate:
        params = f'?fromDate={fromDate}'
    elif toDate:
        params = f'?toDate={toDate}'

    url = f'http://{API_HOST}/api/deployments{params}'
    print(url)
    contents = urllib.request.urlopen(url).read()
    return contents.decode('utf-8')


# Define the tool call for the function
get_deployments_from_api_tool = {
    'type': 'function',
    'function': {
        'name': 'get_deployments_from_api',
        'description': 'Get deployments from an API that have access to all the different deployments happening in the system',
        'parameters': {
            'type': 'object',
            'required': [],
            'properties': {
                'fromDate': {'type': 'string', 'description': 'The start date of the deployments must be in the format ISO 8601 format'},
                'toDate': {'type': 'string', 'description': 'The end date of the deployments must be in the format ISO 8601 format'},
            },
        },
    },
}


def main():
    print("Hello from tool!")
    question = input("Enter your question: ")
    messages = [{'role': 'user', 'content': question} ]
    print('Prompt:', messages[0]['content'])

    available_functions = {
        'get_deployments_from_api': get_deployments_from_api,
    }

    response: ChatResponse = chat(
        OLLAMA_MODEL,
        messages=messages,
        tools=[get_deployments_from_api_tool],
    )

    if response.message.tool_calls:
        # There may be multiple tool calls in the response
        for tool in response.message.tool_calls:
            # Ensure the function is available, and then call it
            if function_to_call := available_functions.get(tool.function.name):
                print('Calling function:', tool.function.name)
                print('Arguments:', tool.function.arguments)
                output = function_to_call(**tool.function.arguments)
                # parse json output
                json_output = json.loads(output)
                
                print('Function output:',json.dumps(json_output, indent=4))
                
            
            else:
                print('Function', tool.function.name, 'not found')

    # Only needed to chat with the model using the tool call results
    if response.message.tool_calls:
        # Add the function response to messages for the model to use
        messages.append(response.message)
        messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})

        # Get final response from model with function outputs
        final_response = chat(OLLAMA_MODEL, messages=messages)
        print('Final response:', final_response.message.content)

    else:
        print('No tool calls returned from model')


if __name__ == "__main__":
    main()
