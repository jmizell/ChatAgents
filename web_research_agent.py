import json
import threading
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


def combine_results(question: str, results: str, new_results: str) -> str:
    msgs = [
        {
            "role": "system",
            "content": f"""Your task is to consolidate and summarize search results for the question: '{question}'.

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


def threaded_filter(question, sliced_results, results, lock):
    for r in sliced_results:
        msgs = [
            {
                "role": "system",
                "content": f"""Please evaluate the following search result based on its relevance to the question: '{question}'.

Title: {r['title']}
URL: {r['href']}
Content: {r['body']}

If the result is relevant, answer with the answer YES, or NO only.
        """
            }
        ]
        response = call_api(msgs, stream=False)
        match = str(response["choices"][0]["message"]
                    ["content"]).lower() == "yes"
        if match:
            with lock:
                results.append(r)
            print(f"Added {r['href']}")


def combine(all_results: list, question: str, query: str, max_results: int = 250) -> str:
    results = []
    lock = threading.Lock()

    num_threads = 4
    slice_size = len(all_results) // num_threads
    threads = []
    for i in range(num_threads):
        start_index = i * slice_size
        end_index = start_index + slice_size
        sliced_results = all_results[start_index:end_index]
        t = threading.Thread(target=threaded_filter, args=(
            question, sliced_results, results, lock))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    response = ""
    combine_input = []
    for r in results:
        combine_input.append(r['body'])
        if token_count("\n".join(combine_input)) > 1000:
            print("processing...")
            response = combine_results(
                question, response, "\n".join(combine_input))
            combine_input = []

    if combine_input:
        print("processing...")
        response = combine_results(
            question, response, "\n".join(combine_input))

    return response


def web_search(question: str, query: str, max_results: int = 250) -> str:
    all_results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, safesearch='off'):
            all_results.append(r)
            if len(all_results) >= max_results:
                break

    return combine(all_results, question, query, max_results)


def wikipedia_search(question: str, query: str, max_results: int = 250) -> str:
    all_results = []
    for page_title in wikipedia.search(query[:300]):
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            continue

        all_results.append({
            "title":page_title,
            "href": wiki_page.url,
            "body": wiki_page.summary,
        })
        if len(all_results) >= max_results:
            break

    return combine(all_results, question, query, max_results)


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
    question = "Can you tell me something interesting about the denver capital?"
    web_result = web_search(question, "denver capital")
    wikipedia_result = wikipedia_search(question, "denver capital")
    print("processing....")
    print(combine_results(question, web_result, wikipedia_result))
