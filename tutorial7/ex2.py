from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from utils.loaders import load_xdan

llm = load_xdan()

template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: 
{instruction}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["instruction"])

llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

if __name__ == '__main__':
    llm_chain.run("What is the capital of England?")
    llm_chain.run("What are alpacas? and how are they different from llamas?")
