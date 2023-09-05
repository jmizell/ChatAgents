import threading
from time import time, sleep

import gradio as gr

from langchain import OpenAI
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory
from langchain.tools import DuckDuckGoSearchResults, PubmedQueryRun, WikipediaQueryRun
from langchain.tools.python.tool import PythonREPLTool
from langchain.utilities import WikipediaAPIWrapper


search = DuckDuckGoSearchResults(num_results=10)
med = PubmedQueryRun()
wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=10))

tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="A wrapper around Duck Duck Go Search. Useful for answering questions about current events. Input should be a search query. Output is a JSON array of the query results."
    ),
    Tool(
        name="Pubmed Search",
        func=med.run,
        description="PubMed is a search engine accessing primarily the MEDLINE database of references and abstracts on life sciences and biomedical topics."
    ),
    Tool(
        name="Wikipedia Search",
        func=wikipedia.run,
        description="Useful for when you need to search Wikipedia directly."
    ),
    Tool(
        name="Python REPL",
        func=PythonREPLTool().run,
        description="A Python shell. Use this to execute Python commands. Input should be a valid Python command. If you want to see the output of a value, you should print it out with `print(...)`."
    ),
]

# Memory options
# https://api.python.langchain.com/en/latest/memory/langchain.memory.buffer.ConversationBufferMemory.html
# https://api.python.langchain.com/en/latest/memory/langchain.memory.summary.ConversationSummaryMemory.html
# https://api.python.langchain.com/en/latest/memory/langchain.memory.combined.CombinedMemory.html
# https://api.python.langchain.com/en/latest/_modules/langchain/memory/token_buffer.html#ConversationTokenBufferMemory
# https://python.langchain.com/docs/modules/memory/multiple_memory
# https://python.langchain.com/docs/modules/memory/agent_with_memory_in_db
# https://python.langchain.com/docs/modules/memory/types/kg
# https://python.langchain.com/docs/modules/memory/types/summary
# https://python.langchain.com/docs/modules/memory/types/buffer_window

memory_llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    llm=memory_llm,
    return_messages=True,
    memory_key="chat_history",
    chat_memory=FileChatMessageHistory("chat_history.txt"),
    verbose=True,
    max_token_limit=1000,
    callbacks=[StdOutCallbackHandler()],
)

llm = OpenAI(temperature=0.7, model_name="gpt-4-0613")
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)


def chat(message):
    try:
        return agent_chain.run(input=message, callbacks=[StdOutCallbackHandler()])
    except Exception as raised_exception:
        error_message = f"Error executing model: \n{raised_exception}"
        print(error_message)
        return error_message


def gradio_bot(message, chat_history):
    return chat(message)


demo = gr.ChatInterface(fn=gradio_bot, cache_examples=False, title="Bot Bot")


if __name__ == "__main__":
    def gradio_thread():
        demo.launch(server_port=4860, server_name="0.0.0.0")


    def user_input_thread():
        while True:
            text = input('\nUSER: ')
            if text == '':
                continue
            resp = chat(text)
            print(f"\nAI: {resp}")
    t1 = threading.Thread(target=gradio_thread)
    t2 = threading.Thread(target=user_input_thread)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
