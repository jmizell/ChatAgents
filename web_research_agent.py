import json
import threading
from datetime import datetime
from time import sleep
import pytz
import openai
import tiktoken
from duckduckgo_search import DDGS
import wikipedia
import requests
from bs4 import BeautifulSoup
import markdownify


FAST_MODEL_NAME = "gpt-3.5-turbo-16k-0613"
FAST_MODEL_MAX_TOKENS = 16385
SMART_MODEL_NAME = "gpt-4-0613"
SMART_MODEL_MAX_TOKENS = 8191


def token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(FAST_MODEL_NAME)
    return len(encoding.encode(string))


def extract_webpage(question: str, url: str, response: str, page_markdown: str) -> str:
    max_retry = 5
    retry = 0

    original_response_length = token_count(response)
    model_name = FAST_MODEL_NAME

    while retry < max_retry:
        msgs = [
            {
                "role": "system",
                "content": f"""Your task is to extract relevant information as bullet points from this web page for the question: '{question}'.
 
Instructions:
1. Combine the information from the 'Current Response' and the 'New Page Segment'.
2. Maintain or extend the length of the 'Current Response'; do not shorten it.
3. Do not omit any details.
4. Include quotes and relevant links.
5. If the 'New Page Segment' does not offer any new information, return the 'Current Response' as is.

Current Response:
{response}

New Page Segment:
{page_markdown}

Please proceed with the task."""
            }
        ]

        new_response = call_api(msgs, stream=False, model_name=model_name)["choices"][0]["message"]["content"]

        if token_count(new_response) >= original_response_length * 0.85:
            return new_response

        print(f"Retry {retry+1}: New response is shorter than original. Retrying for {url}")
        retry += 1
        if retry >= max_retry-1:
            model_name = SMART_MODEL_NAME

    print(f"Max retries reached. Returning original response. Failed segment for {url}")
    return response 


def scrape_and_extract(url: str, question: str) -> str:

    markdown_text, status = scrape_page(url)
    if status != "Success":
        print(f"Scraping failed. Status: {status}")
        return ""

    current_response = ""    
    segment = ""
    segment_token_count = 0
    count = 0
    for word in markdown_text.split(" "):
        tentative_segment = f"{segment} {word}"
        tentative_token_count = token_count(tentative_segment)
        if tentative_token_count <= SMART_MODEL_MAX_TOKENS * 0.5:
            segment = tentative_segment
            segment_token_count = tentative_token_count
        else:
            print(f"Processing page segment {count} for {url}")
            current_response = extract_webpage(question, url, current_response, segment)
            segment = word
            segment_token_count = token_count(segment)
            count=count+1
    if segment:
        print(f"Processing page segment {count} for {url}")
        current_response = extract_webpage(question, url, current_response, segment)

    return current_response


def scrape_page(url: str) -> (str, str):
    """
    Scrape the content of a web page and return it as Markdown-formatted text.

    Parameters:
    - url (str): The URL of the web page to scrape.

    Returns:
    - str: The Markdown-formatted text content of the web page.
    - str: Status message indicating the outcome of the operation ("Success" or an error message).

    Exceptions:
    - Raises requests.exceptions.RequestException for issues like connectivity, timeout, etc.

    Example Usage:
    >>> markdown_text, status = scrape_page('http://www.example.com')
    >>> if status == "Success":
    >>>     print(markdown_text)
    >>> else:
    >>>     print(f"Scraping failed. Status: {status}")
    """

    try:
        response = requests.get(
            url,
            headers={
                'User-Agent': "Mozilla/5.0 (Android 13; Mobile; rv:109.0) Gecko/118.0 Firefox/118.0"
            },
            timeout=20,
        )
        if response.status_code // 100 == 2:
            soup = BeautifulSoup(response.text, 'html.parser')
            markdown_text = markdownify.markdownify(str(soup))
            return markdown_text, "Success"
        else:
            print(f"Failed to fetch the webpage. Status Code: {response.status_code}")
            return "", f"Failed. Status Code: {response.status_code}"
    except requests.exceptions.RequestException as err:
        print(f"An error occurred: {err}")
        return "", f"An error occurred: {err}"


def combine_results(question: str, results: str, new_results: str) -> str:
    max_retry = 5
    retry = 0

    original_response_length = token_count(results)

    while retry < max_retry:
        msgs = [
            {
                "role": "system",
                "content": f"""Your task is to consolidate and summarize search results for the question: '{question}'.

Instructions:
1. Combine the information from the 'Current Result' and the 'New Result'.
2. Do not omit any details from either result.
3. Include links and quotes when relevant.
4. If the 'New Result' does not offer any new information, return the 'Current Result' as is.

Current Result:
{results}

New Result:
{new_results}

Please proceed with the task."""
            }
        ]

        new_response = call_api(msgs, stream=False, model_name=SMART_MODEL_NAME)["choices"][0]["message"]["content"]

        if token_count(new_response) >= original_response_length * 0.85:
            return new_response

        print(f"Retry {retry+1}: New response is shorter than original. Retrying.")
        retry += 1

    print("Max retries reached. Returning original result.")
    return results  # if max retries are reached, return the original result



def threaded_filter(question, sliced_results, results, lock):
    for r in sliced_results:
        msgs = [
            {
                "role": "system",
                "content": f"""Please evaluate the following search result based on its relevance to the question: '{question}'.

Title: {r['title']}
URL: {r['href']}
Content: {r['body']}

Answer with an INTEGER score of the relevencey from 0 through 4. Reply only with 0, 1, 2, 3, or 4.
        """
            }
        ]
        response = call_api(msgs, stream=False)
        score = int(response["choices"][0]["message"]["content"])
        print(f"Score: {score}, URL: {r['href']}")
        if score == 4:
            extracted_info = scrape_and_extract(r['href'], question)
            if token_count(extracted_info) > token_count(r['body']):
                r['body'] = extracted_info
        if score >= 3:
            with lock:
                results.append(r)
            print(f"Added {r['href']}")


def combine(all_results: list, question: str, query: str) -> str:
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
        if token_count("\n".join(combine_input)) > SMART_MODEL_MAX_TOKENS * 0.5:
            print("processing...")
            response = combine_results(question, response, "\n".join(combine_input))
            combine_input = []

    if combine_input:
        print("processing...")
        response = combine_results(
            question, response, "\n".join(combine_input))

    return response


def web_search(question: str, query: str, max_results: int = 250) -> str:
    max_retry = 5
    retry = 0
    all_results = []
    while retry <= max_retry:
        with DDGS() as ddgs:
            all_results = [r for r in ddgs.text(query, region="us-en", safesearch="off", max_results=max_results, backend="lite")]
            if len(all_results) > 0:
                break
        retry+=1

    print(f"Found {len(all_results)} web results to process")
    return combine(all_results, question, query)


def wikipedia_search(question: str, query: str, max_results: int = 250) -> str:
    all_results = []
    for page_title in wikipedia.search(query[:300]):
        try:
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
            continue

        all_results.append({
            "title": page_title,
            "href": wiki_page.url,
            "body": wiki_page.summary,
        })
        if len(all_results) >= max_results:
            break
    print(f"Found {len(all_results)} wikipedia results to process")

    return combine(all_results, question, query)


def call_api(msgs: list[dict], functions: list[dict] = None, stream: bool = True, model_name: str = FAST_MODEL_NAME):
    max_retry = 7
    retry = 0
    while True:
        try:
            create_params = {
                'model': model_name,
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
    question = "I need a list of dispersed camping sites in Colorado"
    print(f"Question: {question}")
    msgs = [
        {
            "role": "system",
            "content": f"""You're an expert web research agent. 
Your goal is to intuit the core interest behind the user's question and generate a web search query that 
captures that interest in the most relevant and specific manner. Consider the nuances of the question 
and provide a tailored query. 

For the question: '{question}', what would be the most appropriate search query? 

Please avoid using quotes around the query."""
        }
    ]
    response = call_api(msgs, stream=False, model_name=SMART_MODEL_NAME)
    search_query = response["choices"][0]["message"]["content"]
    print(f"Search query: {search_query}")
    web_result = web_search(question, search_query, max_results=50)
    msgs = [
        {
            "role": "system",
            "content": f"""Please output a wikipedia search query for this question: {question}. Note: Please provide the query without enclosing it in quotes."""
        }
    ]
    response = call_api(msgs, stream=False, model_name=SMART_MODEL_NAME)
    search_query = response["choices"][0]["message"]["content"]
    print(f"Search query: {search_query}")
    wikipedia_result = wikipedia_search(question, search_query, max_results=50)
    print("processing....")
    print(combine_results(question, web_result, wikipedia_result))
