import os

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader

from utils.loaders import load_dolphin_dpo_laser, load_bge_base_angle_emb

loader = PyPDFDirectoryLoader(path='../resources/test_rag_docs')

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

llm = load_dolphin_dpo_laser()

persist_directory = 'test_rag_docs_pdf'
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
    query = "When Seraphina Celestia Moonshadow was born?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)

    query = "Where Seraphina Celestia Moonshadow was born?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)

    query = "Where did Seraphina venture on her quest for knowledge, and what did she discover there?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)

    query = "Can you elaborate on Seraphina's unique ability and its manifestation during her childhood?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)

    query = "Where did Seraphina venture on her quest for knowledge, and what did she discover there?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)

    query = "What did Seraphina accomplish in her later years that solidified her reputation in Etherialand?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)
