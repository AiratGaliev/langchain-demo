import textwrap

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from utils.loaders import load_xdan_llm

llm = load_xdan_llm()

with open('../resources/test_rag_docs/test_rag.txt') as f:
    about_java = f.read()

fact_extraction_prompt = PromptTemplate(
    input_variables=["text_input"],
    template="Extract the key facts out of this text. Don't include opinions. \
    Give each fact a number and keep them short sentences. :\n\n {text_input}"
)

fact_extraction_chain = LLMChain(llm=llm, prompt=fact_extraction_prompt,
                                 verbose=True)

facts = fact_extraction_chain.run(about_java)

wrapped_text = textwrap.fill(facts,
                             width=100,
                             break_long_words=False,
                             replace_whitespace=False)

if __name__ == '__main__':
    print(len(about_java))
    print(len(wrapped_text))
    print(wrapped_text)
