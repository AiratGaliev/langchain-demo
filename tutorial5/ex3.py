import textwrap

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

from utils.loader import load_xdan_llm

llm = load_xdan_llm()

text_splitter = CharacterTextSplitter()

with open('../resources/about_java.txt') as f:
    about_java = f.read()
texts = text_splitter.split_text(about_java)

docs = [Document(page_content=t) for t in texts[:4]]

prompt_template = """Write a concise bullet point summary of the following:


{text}


CONSCISE SUMMARY IN BULLET POINTS:"""

BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template,
                                     input_variables=["text"])

chain = load_summarize_chain(llm,
                             chain_type="stuff",
                             verbose=True, prompt=BULLET_POINT_PROMPT)

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary,
                             width=100,
                             break_long_words=False,
                             replace_whitespace=False)

if __name__ == '__main__':
    print(len(about_java))
    print(len(wrapped_text))
    print(wrapped_text)
