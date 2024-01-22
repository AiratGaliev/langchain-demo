import os

from crewai import Agent, Task, Crew, Process
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from utils.loaders import load_bge_base_angle_emb, load_openhermes_16k_llm

llm = load_openhermes_16k_llm()

loader = PyPDFDirectoryLoader(path='../resources/test_rag_docs')

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

persist_directory = 'test_rag_docs_pdf'
embedding = load_bge_base_angle_emb()
if not os.path.exists(persist_directory):
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

retriever = vectordb.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


def process_llm_response(llm_response):
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
    print('\n\n')


def qa_chain_result(question: str):
    response = qa_chain(question)
    process_llm_response(response)
    return response["result"]


data_extractor = Tool(
    name="Data Extractor",
    func=qa_chain_result,
    description="""Useful for answer the users question"""
)

assistant = Agent(
    role="Assistant",
    goal="Answer questions",
    backstory="""You are a helpful AI assistant.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[data_extractor]
)


def format_question(question: str) -> str:
    return """{question} You need to use a tool.""".format(question=question)


answer_the_question1 = Task(description=format_question("When Seraphina Celestia Moonshadow was born?"),
                            agent=assistant)

crew = Crew(agents=[assistant], tasks=[answer_the_question1], verbose=2, process=Process.sequential)

if __name__ == '__main__':
    crew.kickoff()
