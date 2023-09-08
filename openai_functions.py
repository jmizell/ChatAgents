import os
import json
from datetime import datetime
import pytz
import openai
import tiktoken
from duckduckgo_search import DDGS


local_tz = pytz.timezone('America/Denver')
utc_now = datetime.now(pytz.utc)
local_time = utc_now.astimezone(local_tz)
model = "gpt-3.5-turbo-16k-0613"
model_tokens = 16000


def token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(string))


def web_search(query: str, max_tokens: int) -> str:

    results = ""
    total_count = 0
    with DDGS() as ddgs:
        for r in ddgs.text(query, safesearch='off'):
            result = f"\nTITLE: {r['title']}\nURL: {r['href']}\n{r['body']}\n"
            total_count = total_count + token_count(result)
            if total_count > max_tokens:
                break        
            results = results + result

    return results

valid_functions = ["web_search"]
functions = [
    {
        "name": "web_search",
        "description": "A wrapper around Duck Duck Go Search. Useful for when you need to answer questions about current events.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "search engine query string",
                },
            },
            "required": ["user_question", "query"],
        },
    },
]

system_message = f"""You're a helpful assistant. You provide concise answer unless prompted for more detail. 
You avoid providing lists, or advice unprompted. Don't make assumptions about what values to plug into functions. 
Ask for clarification if a user request is ambiguous.

Current Date: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')}
"""
messages=[
    {"role": "system", "content": system_message},
]

while True:
    text = input('\nUSER: ')
    if text == '':
        continue
    messages.append({"role": "user", "content": text})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=messages,
        functions=functions,
    )

    if completion['choices'][0]['finish_reason'] == 'function_call':
        function_name = completion['choices'][0]['message']['function_call']['name']
        params = json.loads(completion['choices'][0]['message']['function_call']['arguments'])
        if function_name == "web_search":
            print(f"SYSTEM: searching for '{params['query']}'")
            output = web_search(params['query'], 2000)
        else:
            raise Exception(f"invalid function name {function_name}")

        messages.append({"role": "system", "content": f"function {function_name} output:\n\n{output}"})
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=messages,
            functions=functions,
        )
    
    messages.append(completion['choices'][0]['message'])
    print(f"AI: {completion['choices'][0]['message']['content']}")
