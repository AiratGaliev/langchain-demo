from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory

from utils.loaders import load_openhermes_llm

llm = load_openhermes_llm()

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=40)

conversation_with_summary = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

if __name__ == '__main__':
    conversation_with_summary.predict(input="Hi there! I'm Airat")
    conversation_with_summary.predict(input="I need help with my broken TV")
    conversation_with_summary.predict(input="It makes weird sounds when i turn it on and then goes black")
    conversation_with_summary.predict(input="It seems to be Hardware")
    print(conversation_with_summary.memory.moving_summary_buffer)
