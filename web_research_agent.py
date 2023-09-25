import json
from datetime import datetime
from time import sleep
import pytz
import openai
import tiktoken
from duckduckgo_search import DDGS
import wikipedia


MODEL_NAME = "gpt-3.5-turbo-16k-0613"
MODEL_MAX_TOKENS = 16385


def token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(MODEL_NAME)
    return len(encoding.encode(string))


def combine_results(query: str, results: str, new_results: str) -> str:
    msgs = [
        {
            "role": "system",
            "content": f"""Your task is to consolidate and summarize search results for the query: '{query}'.
            
Instructions:
1. Combine the information from the 'Current Result' and the 'New Result'.
2. Do not omit any details from either result.
3. If the 'New Result' does not offer any new information, return the 'Current Result' as is.

Current Result:
{results}

New Result:
{new_results}

Please proceed with the task."""
        }
    ]
    response = call_api(msgs, stream=False)
    return response["choices"][0]["message"]["content"]



def web_search(query: str, max_results: int = 100) -> str:
    results = ""
    count = 0
    with DDGS() as ddgs:
        for r in ddgs.text(query, safesearch='off'):
            msgs = [
                {
                    "role": "system",
                    "content": f"""Please evaluate the following search result based on its relevance to the query: '{query}'.
                    
Title: {r['title']}
URL: {r['href']}
Content: {r['body']}

If the result is relevant, answer with the answer YES, or NO only.
            """
                }
            ]
            response = call_api(msgs, stream=False)
            match = str(response["choices"][0]["message"]["content"]).lower() == "yes"
            if match:
                results = combine_results(query, results, r['body'])
                print("################################")
                print(results)
                print()
            count = count+1
            if count > max_results:
                break

    print(results)
    return results


def call_api(msgs: list[dict], functions: list[dict] = None, stream: bool = True):
    max_retry = 7
    retry = 0
    while True:
        try:
            create_params = {
                'model': MODEL_NAME,
                'messages': msgs,
                'stream': stream,
            }
            if functions is not None:
                create_params['functions'] = functions
            completion = openai.ChatCompletion.create(**create_params)
            return completion
        except Exception as api_err:
            print(f'\n\nError communicating with OpenAI: "{api_err}"')
            if 'maximum context length' in str(api_err):
                msgs = msgs.pop(0)
                print('\n\n DEBUG: Trimming oldest message')
                continue
            retry += 1
            if retry >= max_retry:
                raise api_err
            print(f'\n\nRetrying in {2 ** (retry - 1) * 5} seconds...')
            sleep(2 ** (retry - 1) * 5)


if __name__ == "__main__":
    web_search("denver capital")
