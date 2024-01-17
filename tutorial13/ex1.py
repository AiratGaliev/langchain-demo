import os
import textwrap

from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from utils.loader import load_dolphin_dpo_laser, load_bge_base_angle_emb

system_prompt = """
Your name is Ash Maurya. You are an expert at Lean Startups.
Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always answer from the perspective of being Ash Maurya."""

llm = load_dolphin_dpo_laser(system_prompt=system_prompt)

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
    response = qa_chain("What is product market fit?")
    process_llm_response(response)
    # response = qa_chain("When should you quit or pivot?")
    # process_llm_response(response)
    # response = qa_chain("What is the purpose of a customer interview?")
    # process_llm_response(response)
    # response = qa_chain("What is your name?")
    # process_llm_response(response)
    # response = qa_chain("What are the customer interviewing techniques?")
    # process_llm_response(response)
    # response = qa_chain("Do you like the color blue?")
    # process_llm_response(response)
    # response = qa_chain("What books did you write?")
    # process_llm_response(response)
