from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from utils.loaders import load_xdan_llm

llm = load_xdan_llm()

with open('../resources/test_rag_docs/test_rag.txt') as f:
    test_rag = f.read()

fact_extraction_prompt = PromptTemplate(
    input_variables=["text_input"],
    template="Extract the key facts out of this text. Don't include opinions. \
    Give each fact a number and keep them short sentences. :\n\n {text_input}"
)

fact_extraction_chain = LLMChain(llm=llm, prompt=fact_extraction_prompt,
                                 verbose=True)

facts = fact_extraction_chain.run(test_rag)

triples_prompt = PromptTemplate(
    input_variables=["facts"],
    template="Take the following list of facts and turn them into triples for a knowledge graph:\n\n {facts}"
)

triples_chain = LLMChain(llm=llm, prompt=triples_prompt, verbose=True)

triples = triples_chain.run(facts)

if __name__ == '__main__':
    print(len(test_rag))
    print(len(triples))
    print(triples)
