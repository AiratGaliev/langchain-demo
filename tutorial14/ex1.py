from crewai import Agent, Task, Process, Crew
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(openai_api_base="http://localhost:1234/v1", openai_api_key="key", temperature=0.0)

researcher = Agent(
    role="Researcher",
    goal="Researcher new AI insights",
    backstory="You are an AI research assistant",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

writer = Agent(
    role="Writer",
    goal="Write compelling and engaging blog posts about AI trends and insights",
    backstory="You are an AI blog post writer who specializes in writing about AI topics",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

task1 = Task(description="Investigate the latest AI trends. Don't need to use a tool.", agent=researcher)
task2 = Task(description="Write a compelling blog post based on the latest AI trends. Don't need to use a tool.",
             agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=2, process=Process.sequential)

if __name__ == '__main__':
    result = crew.kickoff()
    print("######################")
    print(result)
