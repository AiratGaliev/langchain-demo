from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(openai_api_base="http://localhost:1234/v1", openai_api_key="key", temperature=0.0)


def string_length(string: str) -> int:
    return len(string)


calculate_tool = Tool(
    name='String length',
    func=string_length,
    description="Useful to get string length.",
)

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=20,
    return_messages=True
)

conversational_agent = initialize_agent(
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=[calculate_tool],
    llm=llm,
    verbose=True,
    max_iterations=10,
    early_stopping_method='generate',
    memory=memory,
    handle_parsing_errors=True
)

if __name__ == '__main__':
    print(conversational_agent("I want to know this string length: hello world")["output"])
