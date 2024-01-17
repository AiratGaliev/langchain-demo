from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from utils.loader import load_openhermes_llm

llm = load_openhermes_llm()

memory = ConversationBufferWindowMemory(k=2)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory
)

if __name__ == '__main__':
    conversation.predict(input="Hi there! I'm Airat")
    conversation.predict(input="I am looking for some customer support")
    conversation.predict(input="My TV is not working.")
    conversation.predict(input="When I turn it on it makes some weird sounds and then goes black")
    print(conversation.memory.buffer)
