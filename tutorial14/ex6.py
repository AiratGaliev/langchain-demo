import os

from crewai import Agent, Task, Crew, Process
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from utils.loaders import load_bge_base_angle_emb

llm = ChatOpenAI(openai_api_base="http://localhost:1234/v1", openai_api_key="key", temperature=0.0)

loader = DirectoryLoader('../resources/ash_maurya/', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

epub_loader = UnstructuredEPubLoader(
    "../resources/ash_maurya/Running Lean_ Iterate from Plan A to a Plan That Works (Lean Series) - Maurya, Ash.epub")

epub_data = epub_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts_01 = text_splitter.split_documents(documents)
texts_02 = text_splitter.split_documents(epub_data)
texts = texts_01 + texts_02

persist_directory = 'db'
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
    description="""Useful for when you need to extract context data and to answer the users question"""
)

ash_maurya = Agent(
    role="Ash Maurya",
    goal="Always answer from the perspective of being Ash Maurya",
    backstory="""Your name is Ash Maurya. You are an expert at Lean Startups.
    Use the following pieces of context to answer the users question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[data_extractor]
)


def format_question(question: str) -> str:
    return """{question} You need to use a tool to extract context data.""".format(question=question)


answer_the_question1 = Task(description=format_question("What is product market fit?"), agent=ash_maurya)
answer_the_question2 = Task(description=format_question("When should you quit or pivot?"), agent=ash_maurya)
answer_the_question3 = Task(description=format_question("What is the purpose of a customer interview?"),
                            agent=ash_maurya)
answer_the_question4 = Task(description=format_question("What is your name?"), agent=ash_maurya)
answer_the_question5 = Task(description=format_question("What are the customer interviewing techniques?"),
                            agent=ash_maurya)
answer_the_question6 = Task(description=format_question("Do you like the color blue?"), agent=ash_maurya)
answer_the_question7 = Task(description=format_question("What books did you write?"), agent=ash_maurya)

crew = Crew(agents=[ash_maurya], tasks=[answer_the_question1], verbose=2, process=Process.sequential)

if __name__ == '__main__':
    crew.kickoff()
