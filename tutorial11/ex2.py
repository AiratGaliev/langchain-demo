import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from utils.loaders import load_dolphin_dpo_laser, load_bge_base_angle_emb

loader = DirectoryLoader('../resources/new_articles', glob="./*.txt", loader_cls=TextLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

llm = load_dolphin_dpo_laser()

persist_directory = 'db'
embedding = load_bge_base_angle_emb()
if not os.path.exists(persist_directory):
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

retriever = vectordb.as_retriever(search_kwargs={"k": 2})


def process_llm_response(response):
    print('\n\nSources:')
    for source in response["source_documents"]:
        print(source.metadata['source'])
    print('\n\n')


qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

if __name__ == '__main__':
    query = "How much money did Pando raise?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)

    query = "What is the news about Pando?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)

    query = "Who led the round in Pando?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)

    query = "What did databricks acquire?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)

    query = "What is generative ai?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)

    query = "Who is CMA?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)
