from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from utils.loader import load_openhermes

system_prompt = "You are a helpful history professor named Kate."

llm = load_openhermes(system_prompt)

template = """Take the following question: {user_input}

Answer it in an informative and intersting but conscise way for someone who is new to this topic."""

prompt = PromptTemplate(template=template, input_variables=["user_input"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

user_input1 = "When was Marcus Aurelius the emperor of Rome?"
user_input2 = "Who was Marcus Aurelius married to?"

if __name__ == '__main__':
    llm_chain.run(user_input1)
    llm_chain.run(user_input2)
