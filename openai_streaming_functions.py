import json
from datetime import datetime
from time import sleep
import pytz
import openai
import tiktoken
from duckduckgo_search import DDGS
import wikipedia


TIMEZONE = 'America/Denver'
MODEL_NAME = "gpt-3.5-turbo-16k-0613"
MODEL_MAX_TOKENS = 16385
FUNCTIONS = [
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
    {
        "name": "wikipedia_search",
        "description": "A wrapper around Wikipedia Search. Useful for when you need to answer general knowledge qeustions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "wikipedia query string",
                },
            },
            "required": ["user_question", "query"],
        },
    },
]


def token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(MODEL_NAME)
    return len(encoding.encode(string))


def wikipedia_search(query: str, max_tokens: int) -> str:
    results = ""
    total_count = 0
    for page_title in wikipedia.search(query[:300]):

        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
        except (wikipedia.exceptions.PageError,wikipedia.exceptions.DisambiguationError):
            continue

        result = f"\nTITLE: {page_title}\nURL: {wiki_page.url}\n{wiki_page.summary}\n"
        total_count = total_count + token_count(result)
        if total_count > max_tokens:
            break
        results = results + result

    return results


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

    local_time = datetime.now(pytz.utc).astimezone(pytz.timezone(TIMEZONE))
    system_message = {
        "role": "system",
        "content": f"""You're a helpful assistant. You provide concise answer unless prompted for more detail. 
You avoid providing lists, or advice unprompted. Don't make assumptions about what values to plug into functions. 
Ask for clarification if a user request is ambiguous.

Current Date: {local_time.strftime('%Y-%m-%d %H:%M:%S %Z%z')}
    """}

    messages = []
    function_name = ""
    while True:

        if function_name == "":
            text = input('\nUSER: ')
            if text == '':
                continue
            messages.append({"role": "user", "content": text})

        completion = call_api([system_message] + messages, FUNCTIONS, True)
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
                response_text = response_text + \
                    chunk.choices[0].delta.function_call.arguments
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
            if function_name == "wikipedia_search":
                print(f"SYSTEM: searching for '{params['query']}'")
                output = wikipedia_search(params['query'], 1000)
            else:
                raise Exception(f"invalid function name {function_name}")

            messages.append(
                {"role": "function", "name": function_name, "content": output})
