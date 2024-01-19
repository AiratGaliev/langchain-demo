from pprint import pprint

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

from utils.loaders import load_openhermes_llm

llm = load_openhermes_llm()

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm)
)

if __name__ == '__main__':
    conversation.predict(input="Hi I am Airat. My TV is broken but it is under warranty.")
    conversation.predict(input="How can I get it fixed. The warranty number is A512453")
    conversation.predict(input="Can you send the repair person call Dave to fix it?.")
    pprint(conversation.memory.entity_store.store)
