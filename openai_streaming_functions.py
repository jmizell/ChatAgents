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
model = "gpt-4-0613"
model_tokens = 8192


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

function_name = ""
while True:

    if function_name == "":
        text = input('\nUSER: ')
        if text == '':
            continue
        messages.append({"role": "user", "content": text})

    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        functions=functions,
        stream=True,
    )
    response_text = ""
    function_name = ""
    first_line = True
    for chunk in completion:
        # print(chunk)
        if chunk.choices[0].finish_reason:
            break

        if "function_call" in chunk.choices[0].delta:
            if "name" in chunk.choices[0].delta.function_call:
                function_name = chunk.choices[0].delta.function_call.name
            response_text = response_text + chunk.choices[0].delta.function_call.arguments
        if "content" in chunk.choices[0].delta and chunk.choices[0].delta.content:
            response_text = response_text + chunk.choices[0].delta.content
            if first_line:
                print("AI: ", end="")
                first_line = False
            print(chunk.choices[0].delta.content, end="")

        if not function_name == "" and first_line:
            print(f"SYTEM: function call {function_name}")
            first_line = False

    if function_name == "":
        messages.append({"role": "assistant", "content": response_text})
    else:
        params = json.loads(response_text)
        if function_name == "web_search":
            print(f"SYSTEM: searching for '{params['query']}'")
            output = web_search(params['query'], 1000)
        else:
            raise Exception(f"invalid function name {function_name}")
    
        messages.append({"role": "function", "name": function_name, "content": output})
    
