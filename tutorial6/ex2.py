from langchain.chains import LLMSummarizationCheckerChain

from utils.loaders import load_xdan_llm

llm = load_xdan_llm()

with open('../resources/test_rag_docs/test_rag.txt') as f:
    about_java = f.read()

checker_chain = LLMSummarizationCheckerChain.from_llm(llm=llm, verbose=True, max_checks=2)

final_summary = checker_chain.run(about_java)

if __name__ == '__main__':
    print(len(about_java))
    print(len(final_summary))
    print(final_summary)
