from crewai import Agent, Task, Crew, Process
from langchain.agents import Tool

from utils.loaders import load_dolphin_llm

llm = load_dolphin_llm()


def string_length(string: str) -> int:
    return len(string)


string_tool = Tool(
    name='String tool',
    func=string_length,
    description="Useful to work with strings.",
)

agent = Agent(
    role="Custom Agent",
    goal="Use tools",
    backstory="Expert in the use of tools",
    verbose=True,
    llm=llm,
    tools=[string_tool]
)

task1 = Task(
    description="I want to know the result of passing this string: hello world",
    agent=agent
)

crew = Crew(agents=[agent], tasks=[task1], verbose=2, process=Process.sequential)

if __name__ == '__main__':
    result = crew.kickoff()
    print("######################")
    print(result)
