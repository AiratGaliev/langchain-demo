import os
import textwrap

from langchain.chains import RetrievalQA
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from utils.loaders import load_dolphin_dpo_laser, load_bge_base_angle_emb

llm = load_dolphin_dpo_laser()

loader = PyPDFDirectoryLoader(path='../resources/new_papers')

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

persist_directory = 'db'
embedding = load_bge_base_angle_emb()
if not os.path.exists(persist_directory):
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

retriever = vectordb.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
    print('\n\n')


if __name__ == '__main__':
    response = qa_chain("What is Flash attention?")
    process_llm_response(response)
    response = qa_chain("What does IO-aware mean?")
    process_llm_response(response)
    response = qa_chain("What is tiling in flash-attention?")
    process_llm_response(response)
    response = qa_chain("What is toolformer?")
    process_llm_response(response)
    response = qa_chain("What tools can be used with toolformer?")
    process_llm_response(response)
    response = qa_chain("How many examples do we need to provide for each tool?")
    process_llm_response(response)
    response = qa_chain("What are the best retrieval augmentations for LLMs?")
    process_llm_response(response)
    response = qa_chain("What are the differences between REALM and RAG?")
    process_llm_response(response)
