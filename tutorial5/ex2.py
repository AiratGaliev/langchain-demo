import textwrap

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

from utils.loaders import load_xdan_llm

llm = load_xdan_llm()

text_splitter = CharacterTextSplitter()

with open('../resources/test_rag_docs/test_rag.txt') as f:
    test_rag = f.read()
texts = text_splitter.split_text(test_rag)

docs = [Document(page_content=t) for t in texts[:4]]

chain = load_summarize_chain(llm,
                             chain_type="stuff",
                             verbose=True)

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary,
                             width=100,
                             break_long_words=False,
                             replace_whitespace=False)

if __name__ == '__main__':
    print(len(test_rag))
    print(len(wrapped_text))
    print(wrapped_text)
