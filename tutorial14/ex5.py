from crewai import Agent, Task, Crew, Process
from langchain.tools import tool

from utils.loaders import load_dolphin_llm

llm = load_dolphin_llm()


@tool
def calculator_tool(numbers: str) -> int:
    """Useful for when you need to answer questions about math."""
    numbers = numbers.split(",")
    return int(numbers[0]) * int(numbers[1])


calculator = Agent(
    role="Calculator",
    goal="Use tools",
    backstory="Expert in the use of tools",
    verbose=True,
    llm=llm,
    tools=[calculator_tool]
)

calculate = Task(
    description="Pass two numbers 3 and 10 to the calculator separated by a comma. Give the result.",
    agent=calculator
)

crew = Crew(agents=[calculator], tasks=[calculate], verbose=2, process=Process.sequential)

if __name__ == '__main__':
    result = crew.kickoff()
    print("######################")
    print(result)
