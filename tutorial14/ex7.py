from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(openai_api_base="http://localhost:1234/v1", openai_api_key="key", temperature=0.0)

manager = Agent(
    role="manager",
    goal="Manage cold email writing for direct to consumer brands",
    backstory="You manage writing cold emails by writer and critic",
    verbose=True,
    allow_delegation=True,
    memory=True,
    llm=llm
)

writer = Agent(
    role="writer",
    goal="Discover best practices for writing cold emails headlines",
    backstory="""You're a world class writer working on a major cold email campaign, promoting a video editing
     solution""",
    verbose=True,
    allow_delegation=False,
    memory=True,
    llm=llm
)

critic = Agent(
    role="critic",
    goal="Critique the emails, pretending to be a potential customer a busy CMO of direct to consumer company",
    backstory="""You're a busy CMO of a direct to consumer company, and you're looking for a video editing solution to
     help you create more video ads for your products but you hate getting cold emails and don't trust them""",
    verbose=True,
    allow_delegation=False,
    memory=True,
    llm=llm
)

task1 = Task(description="""Write 3 short cold emails. Emails should be full-fledged and don't write short 
examples, unnecessary text that is irrelevant to the emails. Emails should promote video editing solutions for 
direct-to-consumer brands that spend at least $3k per day on Facebook ads. Don't need to use a tool.""", agent=writer)

task2 = Task(description="""Check 3 emails written by the writer and write a detailed critique for each of them. 
Require that emails be full-fledged emails, not sample emails content or outlines. 
Explain the proper way to write emails.
Require writing all 3 emails, not a discussion of the critique. Don't need to use a tool.""", agent=critic)

task3 = Task(description="""Rewrite previously written emails after critic. Write only emails, not discuss criticism. 
Don't need to use a tool.""", agent=writer)

crew = Crew(agents=[writer, critic], tasks=[task1, task2, task3], verbose=True,
            process=Process.sequential)

if __name__ == '__main__':
    result = crew.kickoff()
    print("######################")
    print(result)
