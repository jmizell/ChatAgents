from time import time, sleep
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationSummaryBufferMemory
from langchain import OpenAI
from langchain.tools import DuckDuckGoSearchResults
from langchain.agents import initialize_agent
from langchain.tools import PubmedQueryRun
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.memory.chat_message_histories import FileChatMessageHistory
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain import LLMMathChain

llm = OpenAI(temperature=0)
search = DuckDuckGoSearchResults(num_results=10)
med = PubmedQueryRun()
wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=10))
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="A wrapper around Duck Duck Go Search. Useful for when you need to answer questions about current events. Input should be a search query. Output is a JSON array of the query results"
    ),
    Tool(
        name="Pubmed Search",
        func=med.run,
        description="PubMed is aWW search engine accessing primarily the MEDLINE database of references and abstracts on life sciences and biomedical topics."
    ),
    Tool(
        name="Wikipedia Search",
        func=wikipedia.run,
        description="useful for when you need to search wikipedia directly"
    ),
    Tool(
        name="Python REPL",
        func=PythonREPLTool().run,
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`."
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
    WriteFileTool(),
    ReadFileTool(),
]

model = ChatOpenAI(temperature=1, model="gpt-3.5-turbo-16k")
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

print(agent.run("Find some dirt trails that allow onewheels around 60 miles from Denver Colorado."))
