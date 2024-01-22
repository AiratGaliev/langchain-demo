import os
import textwrap

from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from utils.loaders import load_dolphin_dpo_laser, load_bge_base_angle_emb

system_prompt = """
You are a helpful AI assistant.
Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer."""

llm = load_dolphin_dpo_laser(system_prompt=system_prompt)

loader = DirectoryLoader('../resources/test_rag_docs', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

epub_loader = UnstructuredEPubLoader("../resources/test_rag_docs/test_rag.epub")

epub_data = epub_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts_01 = text_splitter.split_documents(documents)
texts_02 = text_splitter.split_documents(epub_data)
texts = texts_01 + texts_02

persist_directory = 'test_rag_docs_db'
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
    response = qa_chain(
        "Who was Seraphina Celestia Moonshadow, and what was her role at the Astral Academy?")
    process_llm_response(response)
    response = qa_chain(
        "How did Orion Emberheart become associated with Seraphina, and what was his area of magical focus?")
    process_llm_response(response)
    response = qa_chain("What advice did Seraphina give to Luna Silverbreeze regarding harmonizing empathic abilities with the natural elements?")
    process_llm_response(response)
    response = qa_chain(
        "Who were Zephyr and Aether, and how did Seraphina interact with them?")
    process_llm_response(response)
    response = qa_chain(
        "What insights did Seraphina share with Quilliana Stellargaze about the forgotten chronicles of Etherialand?")
    process_llm_response(response)
    response = qa_chain(
        "Who was Xaloxian the Starbound, and what did he inquire about from Seraphina?")
    process_llm_response(response)
    response = qa_chain(
        "In the twilight of her years, where did Seraphina journey and who did she seek counsel from?")
    process_llm_response(response)
    response = qa_chain(
        "What did Aionis reveal to Seraphina about the tapestry of time and her life's purpose?")
    process_llm_response(response)
    response = qa_chain(
        "What did Seraphina discover in the Nexus of Possibilities, and how did it impact her understanding of her life's journey?")
    process_llm_response(response)
    response = qa_chain(
        "How did Seraphina conclude her journey at the Celestial Gala, and what message did she share with the gathered multitude?")
    process_llm_response(response)
    response = qa_chain(
        "What happened to Seraphina after the Celestial Gala, and what was the enduring legacy she left behind?")
    process_llm_response(response)
