import random

from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools.web_page import WebPageTool
from utils.loaders import load_dolphin_dpo_laser

fixed_prompt = '''Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant doesn't know anything about random numbers or anything related to the meaning of life and should use a tool for questions about these topics.

Assistant also doesn't know information about content on webpages and should always check if asked.

Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.'''

llm = load_dolphin_dpo_laser(system_prompt=fixed_prompt)

page_getter = WebPageTool()


def meaning_of_life(input=""):
    return 'The meaning of life is 42 if rounded but is actually 42.17658'


life_tool = Tool(name='MOL', func=meaning_of_life,
                 description="Useful for when you need to answer questions about the meaning of life. input should be MOL ")


def random_num(input=""):
    return random.randint(0, 5)


random_tool = Tool(name='random', func=random_num,
                   description="Useful for when you need to get a random number. input should be 'random'")

tools = [page_getter, random_tool, life_tool]

memory = ConversationBufferWindowMemory(memory_key='chat_history', k=3, return_messages=True)

conversational_agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    memory=memory,
    handle_parsing_errors=True
)

if __name__ == '__main__':
    conversational_agent.run("Is there an article about home robot on https://techcrunch.com/?")
    conversational_agent.run("What are the titles of the top stories on www.cbsnews.com/?")
