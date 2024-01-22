import textwrap

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

from utils.loaders import load_xdan_llm

llm = load_xdan_llm()

text_splitter = CharacterTextSplitter()

with open('../resources/test_rag_docs/test_rag.txt') as f:
    about_java = f.read()
texts = text_splitter.split_text(about_java)

docs = [Document(page_content=t) for t in texts[:4]]

chain = load_summarize_chain(llm, verbose=True, chain_type="refine")

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print(wrapped_text)

if __name__ == '__main__':
    print(len(about_java))
    print(len(wrapped_text))
    print(wrapped_text)
