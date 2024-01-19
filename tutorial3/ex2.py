from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory

from utils.loaders import load_openhermes_llm

llm = load_openhermes_llm()

memory = ConversationSummaryMemory(llm=llm)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory
)

if __name__ == '__main__':
    conversation.predict(input="Hi there! I'm Airat")
    conversation.predict(input="How are you today?")
    conversation.predict(input="Can you help me with some customer support?")
    print(conversation.memory.buffer)
