from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

from utils.loaders import load_dolphin

llm = load_dolphin()

examples = [
    {"word": "Happy", "antonym": "Sad"},
    {"word": "tall", "antonym": "short"},
]

example_formatter_template = """
Word: {word}
Antonym: {antonym}\n
"""
example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
)

few_shot_prompt = FewShotPromptTemplate(
    # These are the examples we want to insert into the prompt.
    examples=examples,
    # This is how we want to format the examples when we insert them into the prompt.
    example_prompt=example_prompt,
    # The prefix is some text that goes before the examples in the prompt.
    # Usually, this consists of intructions.
    prefix="Give only the antonym of every input",
    # The suffix is some text that goes after the examples in the prompt.
    # Usually, this is where the user input will go
    suffix="Word: {input}\nAntonym: ",
    # The input variables are the variables that the overall prompt expects.
    input_variables=["input"],
    # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
    example_separator="\n"
)

chain = LLMChain(llm=llm, prompt=few_shot_prompt, verbose=True)

if __name__ == '__main__':
    print(chain.run("Big"))
