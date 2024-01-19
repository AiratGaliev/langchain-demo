from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from utils.loaders import load_openhermes

llm = load_openhermes()

restaurant_template = """
I want you to act as a naming consultant for new restaurants.

Return only a list of restaurant names. Each name should be short, catchy and easy to remember. It should relate to the type of restaurant you are naming.

What are some good names for a restaurant that is {restaurant_description}?
"""

prompt = PromptTemplate(
    input_variables=["restaurant_description"],
    template=restaurant_template,
)

description = "a Greek place that serves fresh lamb souvlakis and other Greek food "
description_02 = "a burger place that is themed with baseball memorabilia"
description_03 = "a cafe that has live hard rock music and memorabilia"

prompt.format(restaurant_description=description)

chain = LLMChain(llm=llm, prompt=prompt)

if __name__ == '__main__':
    chain.run(description)
