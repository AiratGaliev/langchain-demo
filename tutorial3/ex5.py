from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationKGMemory
from langchain.prompts.prompt import PromptTemplate

from utils.loader import load_openhermes_llm

llm = load_openhermes_llm()

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""

prompt = PromptTemplate(
    input_variables=["history", "input"], template=template
)

conversation_with_kg = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=prompt,
    memory=ConversationKGMemory(llm=llm)
)

if __name__ == '__main__':
    conversation_with_kg.predict(input="Hi there! I'm Airat")
    conversation_with_kg.predict(input="My TV is broken and I need some customer assistance")
    conversation_with_kg.predict(input="It makes weird sounds when i turn it on and then goes black")
    conversation_with_kg.predict(input="Yes it is and it is still under warranty. my warranty number is A512423")
    print(conversation_with_kg.memory.kg)
    print(conversation_with_kg.memory.kg.get_triples())
