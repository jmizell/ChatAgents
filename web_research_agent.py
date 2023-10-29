import json
import os
import time
import re
import threading
import argparse
import json
from datetime import datetime
from time import sleep
import pytz
import openai
import tiktoken
from duckduckgo_search import DDGS
import wikipedia
import requests
from bs4 import BeautifulSoup


FAST_MODEL_NAME = os.getenv("FAST_MODEL_NAME", "gpt-3.5-turbo-16k-0613")
FAST_MODEL_MAX_TOKENS = int(os.getenv("FAST_MODEL_MAX_TOKENS", "16385"))
SMART_MODEL_NAME = os.getenv("SMART_MODEL_NAME", "gpt-4-0613")
SMART_MODEL_MAX_TOKENS = int(os.getenv("SMART_MODEL_MAX_TOKENS", "8191"))


def html_to_markdown(html):
    soup = BeautifulSoup(html, 'html.parser')
    markdown_text = ""

    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'ul', 'ol', 'li', 'p', 'br']):
        if tag.name == 'h1':
            markdown_text += '# {}\n'.format(tag.text.replace('\n', ' ').strip())
        elif tag.name == 'h2':
            markdown_text += '## {}\n'.format(tag.text.replace('\n', ' ').strip())
        elif tag.name == 'h3':
            markdown_text += '### {}\n'.format(tag.text.replace('\n', ' ').strip())
        elif tag.name == 'h4':
            markdown_text += '#### {}\n'.format(tag.text.replace('\n', ' ').strip())
        elif tag.name == 'h5':
            markdown_text += '##### {}\n'.format(tag.text.replace('\n', ' ').strip())
        elif tag.name == 'h6':
            markdown_text += '###### {}\n'.format(tag.text.replace('\n', ' ').strip())
        elif tag.name == 'a':
            markdown_text += '[{}]({})\n'.format(tag.text.replace('\n', ' ').strip(), tag.get('href', '').replace('\n', ' ').strip())
        elif tag.name == 'ul':
            for li in tag.find_all('li'):
                markdown_text += '* {}\n'.format(li.text.replace('\n', ' ').strip())
        elif tag.name == 'ol':
            counter = 1
            for li in tag.find_all('li'):
                markdown_text += '{}. {}\n'.format(counter, li.text.replace('\n', ' ').strip())
                counter += 1
        elif tag.name == 'p':
            markdown_text += '{}\n'.format(tag.text.replace('\n', ' ').strip())
        elif tag.name == 'br':
            markdown_text += '\n'
        else:
            markdown_text += tag.text.replace('\n', ' ').strip()

        markdown_text = markdown_text.strip() + '\n\n'

    markdown_text = re.sub(r' +', ' ', markdown_text)
    return markdown_text


def save_to_file(filename, content):
    """Saves content to a specified filename."""
    with open(filename, "w") as file:
        file.write(content)


def count_tokens(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(string))


def extract_information_from_page(question: str, url: str, response: str, page_markdown: str) -> str:
    """Extract relevant information from a webpage and combine it with an existing response."""
    max_retry = 2
    retry = 0

    original_response_length = count_tokens(response)
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
6. Please begin your response with the consolidated information without referencing the instructions.

Current Response:
{response}

New Page Segment:
{page_markdown}

Please proceed with the task."""
            }
        ]

        new_response = interact_with_openai_api(msgs, stream=False, model_name=model_name)[
            "choices"][0]["message"]["content"]

        if count_tokens(new_response) >= original_response_length * 0.85:
            return new_response

        print(
            f"Retry {retry+1}: New response is shorter than original. Retrying for {url}")
        retry += 1
        if retry > 1:
            model_name = SMART_MODEL_NAME

    print(
        f"Max retries reached. Returning original response. Failed segment for {url}")
    return response


def extract_from_url(url: str, question: str) -> str:
    """Scrape content from a URL and extract relevant information."""

    markdown_text, status = scrape_content_from_url(url)
    if status != "Success":
        print(f"Scraping failed. Status: {status}")
        return ""

    current_response = ""
    segment = ""
    segment_token_count = 0
    count = 0
    for word in markdown_text.split(" "):
        tentative_segment = f"{segment} {word}"
        tentative_token_count = count_tokens(tentative_segment)
        if tentative_token_count <= SMART_MODEL_MAX_TOKENS * 0.5:
            segment = tentative_segment
            segment_token_count = tentative_token_count
        else:
            print(f"Processing page segment {count} for {url}")
            current_response = extract_information_from_page(
                question, url, current_response, segment)
            segment = word
            segment_token_count = count_tokens(segment)
            count = count+1
    if segment:
        print(f"Processing page segment {count} for {url}")
        current_response = extract_information_from_page(
            question, url, current_response, segment)

    return current_response


def scrape_content_from_url(url: str) -> (str, str):
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
    >>> text, status = scrape_page('http://www.example.com')
    >>> if status == "Success":
    >>>     print(text)
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
            markdown_text = html_to_markdown(response.text)
            return markdown_text, "Success"
        else:
            print(
                f"Failed to fetch the webpage {url}. Status Code: {response.status_code}")
            return "", f"Failed. Status Code: {response.status_code}"
    except requests.exceptions.RequestException as err:
        print(f"An error occurred: {err}")
        return "", f"An error occurred: {err}"


def consolidate_search_results(question: str, results: str, new_results: str) -> str:
    """Combine and summarize multiple search results into one cohesive response."""

    max_retry = 2
    retry = 0

    original_response_length = count_tokens(results)
    model_name = FAST_MODEL_NAME

    while retry < max_retry:
        msgs = [
            {
                "role": "system",
                "content": f"""Your task is to consolidate and summarize search results for the question: '{question}'.

Instructions:
1. Combine the information from the 'Current Result' and the 'New Result'.
2. Do not omit any details from either result.
3. Include references, quotes and relevant links.
4. If the 'New Result' does not offer any new information, return the 'Current Result' as is.
5. Please begin your response with the consolidated information without referencing the instructions.

Current Result:
{results}

New Result:
{new_results}

Please proceed with the task."""
            }
        ]

        new_response = interact_with_openai_api(msgs, stream=False, model_name=model_name)[
            "choices"][0]["message"]["content"]

        if count_tokens(new_response) >= original_response_length * 0.85:
            return new_response

        print(
            f"Retry {retry+1}: New response is shorter than original. Retrying.")
        retry += 1
        if retry > 1:
            model_name = SMART_MODEL_NAME

    print("Max retries reached. Returning original result.")
    return results  # if max retries are reached, return the original result


def get_score(response: str) -> int:
    first_line = response.split('\n')[0]
    match = re.search(r'\b\d+\b', first_line)
    if match:
        return int(match.group())
    raise ValueError("No integer value found in the response.")


def filter_results_in_threads(question, sliced_results, results, lock):
    """Evaluate search results in parallel threads based on their relevance."""
    for r in sliced_results:
        msgs = [
            {
                "role": "system",
                "content": f"""Task Description:

You are a Search Result Relevancy Evaluator. Given a set of text inputs that represent the title, URL, and content of 
a search result, you operate as a function that must compute and output an integer score that quantifies the search 
result's relevance to a specific user query.

Task Requirements:
- Evaluate the given search result in the context of its relevance to a predetermined user query.
- Output the evaluation as an INTEGER value that falls within the inclusive range of 0 to 4.
- STRICTLY output ONLY one of the following integer values: 0, 1, 2, 3, or 4.
- Outputting any value other than the specified integers will trigger a runtime exception and consequently crash the program.

Evaluation Task:

Given the search result as follows, please assess its relevance to the query '{question}'.

  Title: {r['title']}
  URL: {r['href']}
  Content: {r['body']}

Response Instructions:

Return an INTEGER score representing the search result's relevance to the query. The score must strictly be one of the 
following: 0, 1, 2, 3, or 4. Any other type of response will result in a runtime exception and will terminate the program.

Response:
Score: """
            }
        ]

        response = interact_with_openai_api(msgs, stream=False)
        score = get_score(response["choices"][0]["message"]["content"])
        print(f"Score: {score}, URL: {r['href']}")
        if score == 4:
            extracted_info = extract_from_url(r['href'], question)
            if count_tokens(extracted_info) > count_tokens(r['body']):
                r['body'] = extracted_info
        if score >= 3:
            with lock:
                results.append(r)
            print(f"Added {r['href']}")


def consolidate_results(all_results: list, question: str, query: str) -> str:
    """Combine all relevant search results into a single response."""

    results = []
    lock = threading.Lock()

    num_threads = 4
    if num_threads > len(all_results):
        num_threads = len(all_results)
    slice_size = len(all_results) // num_threads
    threads = []
    for i in range(num_threads):
        start_index = i * slice_size
        end_index = start_index + slice_size
        sliced_results = all_results[start_index:end_index]
        t = threading.Thread(target=filter_results_in_threads, args=(question, sliced_results, results, lock))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    response = ""
    combine_input = []
    for r in results:
        combine_input.append(
            f"\n\nTitle: {r['title']}\nURL: {r['href']}\nContent: {r['body']}")
        if count_tokens("\n".join(combine_input)) > SMART_MODEL_MAX_TOKENS * 0.5:
            print("processing...")
            response = consolidate_search_results(
                question, response, "\n".join(combine_input))
            combine_input = []

    if combine_input:
        print("processing...")
        response = consolidate_search_results(
            question, response, "\n".join(combine_input))

    return response


def web_search(question: str, query: str, max_results: int = 250) -> str:
    """Conduct a web search and return a consolidated result."""

    max_retry = 5
    retry = 0
    all_results = []
    while retry <= max_retry:
        with DDGS() as ddgs:
            all_results = [r for r in ddgs.text(
                query, region="us-en", safesearch="off", max_results=max_results, backend="lite")]
            if len(all_results) > 0:
                break
        retry += 1

    print(f"Found {len(all_results)} web results to process")
    return consolidate_results(all_results, question, query)


def execute_wikipedia_search(question: str, query: str, max_results: int = 250) -> str:
    """Search Wikipedia for relevant articles and return a consolidated result."""
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

    return consolidate_results(all_results, question, query)


def interact_with_openai_api(msgs: list[dict], functions: list[dict] = None, stream: bool = True, model_name: str = FAST_MODEL_NAME):
    """Call the OpenAI API and handle any potential errors."""
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
            completion = openai.ChatCompletion.create(
                headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "localhost",
                },
                **create_params)
            if not stream and bool(os.getenv("DEBUG", "")):
                print(json.dumps(msgs, indent=2, sort_keys=True))
                print(json.dumps(completion, indent=2, sort_keys=True))
            return completion
        except Exception as api_err:
            print(f'\n\nError communicating with OpenAI: "{api_err}"')
            retry += 1
            if retry >= max_retry:
                raise api_err
            print(f'\n\nRetrying in {2 ** (retry - 1) * 5} seconds...')
            sleep(2 ** (retry - 1) * 10)


def _start():
    """
    Initiates the main execution of the program.

    This function performs the following tasks:
    1. Defines a question regarding dispersed camping sites in Utah.
    2. Queries an expert web research agent to generate a relevant search query.
    3. Executes a web search using the generated query.
    4. Generates a Wikipedia-specific query for the same question.
    5. Searches Wikipedia using the generated query.
    6. Consolidates the results from the web and Wikipedia searches and prints them.

    Note: This function is intended for internal use and should not be imported or called externally.
    """

    parser = argparse.ArgumentParser(
        description="Search the web and Wikipedia based on user query.")
    parser.add_argument("--web", type=int,
                        help="Number of web search results", default=10)
    parser.add_argument("--wiki", type=int,
                        help="Number of Wikipedia search results", default=0)
    args = parser.parse_args()

    print(f"Fast model {FAST_MODEL_NAME}:{FAST_MODEL_MAX_TOKENS}")
    print(f"Smart model {SMART_MODEL_NAME}:{SMART_MODEL_MAX_TOKENS}")

    while True:
        question = input("Enter your question: ")
        if question != "":
            break

    web_result = ""
    if args.web > 0:
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
        response = interact_with_openai_api(
            msgs, stream=False, model_name=SMART_MODEL_NAME)
        search_query = response["choices"][0]["message"]["content"]
        print(f"Search query: {search_query}")
        web_result = web_search(question, search_query, max_results=args.web)

    wikipedia_result = ""
    if args.wiki > 0:
        msgs = [
            {
                "role": "system",
                "content": f"""You're tasked with generating a query for Wikipedia, an encyclopedia. Think of topics or subject areas that Wikipedia is likely to have comprehensive articles on. The query should be tailored to retrieve relevant and in-depth information on the topic at hand.

Considering the question: '{question}', what would be the most encyclopedic search query for Wikipedia? 

Please provide the query without enclosing it in quotes."""
            }
        ]
        response = interact_with_openai_api(
            msgs, stream=False, model_name=SMART_MODEL_NAME)
        search_query = response["choices"][0]["message"]["content"]
        wikipedia_result = execute_wikipedia_search(
            question, search_query, max_results=args.wiki)

    result = web_result
    if len(wikipedia_result) > 0 and len(web_result) > 0:
        result = consolidate_search_results(
            question, web_result, wikipedia_result)
    elif len(wikipedia_result) > 0:
        result = wikipedia_result

    print("Search complete\n\n")
    print(result)
    msgs = [
        {
            "role": "system",
            "content": f"""Provide a file name in snake case, ending in .txt, for to save the answer to this user question: '{question}'."""
        }
    ]
    response = interact_with_openai_api(
        msgs, stream=False, model_name=SMART_MODEL_NAME)
    file_name = response["choices"][0]["message"]["content"]
    save_to_file(f"research_result_{int(time.time())}_{file_name}", result)


if __name__ == "__main__":
    _start()
