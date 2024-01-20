from crewai import Agent, Task, Crew, Process
from langchain.agents import Tool

from utils.loaders import load_dolphin_llm

llm = load_dolphin_llm()


def string_length(string: str) -> int:
    return len(string)


calculate_tool = Tool(
    name='String length',
    func=string_length,
    description="Useful to get string length.",
)


calculator = Agent(
    role="Calculator",
    goal="Use tools",
    backstory="Expert, that doesn't know anything but uses tools",
    verbose=True,
    llm=llm,
    tools=[calculate_tool]
)

task1 = Task(
    description="I want to know this string length: hello world",
    agent=calculator
)

crew = Crew(agents=[calculator], tasks=[task1], verbose=2, process=Process.sequential)

if __name__ == '__main__':
    result = crew.kickoff()
    print("######################")
    print(result)
