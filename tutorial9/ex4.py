from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import PythonREPL
from langchain.utilities import WikipediaAPIWrapper

from utils.loaders import load_openhermes

llm = load_openhermes()
python_repl = PythonREPL()
wikipedia = WikipediaAPIWrapper()
search = DuckDuckGoSearchAPIWrapper()

tools = [
    Tool(
        name="python repl",
        func=python_repl.run,
        description="Useful for when you need to use python to answer a question. You should input python code"
    ),
    Tool(
        name='DuckDuckGo Search',
        func=search.run,
        description="Useful for when you need to do a search on the internet to find information that another tool can't find. be specific with your input."
    ),
    Tool(
        name='wikipedia',
        func=wikipedia.run,
        description="Useful for when you need to look up a topic, country or person on wikipedia"
    )
]

zero_shot_agent = initialize_agent(
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    max_iterations=3,
    verbose=True,
    handle_parsing_errors=True
)

if __name__ == '__main__':
    print(zero_shot_agent.run("When was Barak Obama born?"))
    print(zero_shot_agent.run("What is 17*6?"))
    print(zero_shot_agent.run("Tell me about LangChain"))
    print(zero_shot_agent.run("Tell me about Singapore"))
    print(zero_shot_agent.run("what is the current price of btc"))
    print(zero_shot_agent.run("Is 11 a prime number?"))
    print(zero_shot_agent.run("Write a function to check if 11 a prime number and test it"))
