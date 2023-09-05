from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
import faiss
from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchResults
from langchain.tools import PubmedQueryRun
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools.python.tool import PythonREPLTool


search = DuckDuckGoSearchResults(num_results=10)
med = PubmedQueryRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=10))
tools = [
    Tool(
        name="search",
        func=search.run,
        description="A wrapper around Duck Duck Go Search. Useful for when you need to answer questions about current events. Input should be a search query. Output is a JSON array of the query results"
    ),
    WriteFileTool(),
    ReadFileTool(),
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
]

# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=1, model="gpt-3.5-turbo-16k"),
    memory=vectorstore.as_retriever(),
)
# Set verbose to be true
agent.chain.verbose = True

print(agent.run(["Write a blog post listing dirt trails that allow onewheels around 60 miles from Denver Colorado."]))