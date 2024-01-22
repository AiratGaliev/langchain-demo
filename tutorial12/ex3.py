import os

from langchain.chains import RetrievalQA
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from utils.loaders import load_openhermes_llm, load_bge_base_angle_emb

llm = load_openhermes_llm()

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

rqa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


def process_llm_response(llm_response):
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata)
    print('\n\n')


if __name__ == '__main__':
    response = rqa("When Seraphina Celestia Moonshadow was born?")
    process_llm_response(response)
