from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

from utils.loaders import load_openhermes_16k_llm, load_bge_base_angle_emb

embedding = load_bge_base_angle_emb()

llm = load_openhermes_16k_llm()

loader = PyPDFLoader(file_path='../resources/test_rag_docs/test_rag.pdf')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

documents = loader.load_and_split(text_splitter=text_splitter)

docsearch = FAISS.from_documents(documents, embedding)
retriever = docsearch.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

if __name__ == '__main__':
    qa_chain("Where Seraphina Celestia Moonshadow was born?")
