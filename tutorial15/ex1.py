# it doesn't work, but just example how to implement it with autogen and openai
import os

import autogen
from autogen import config_list_from_json
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAI

from utils.loaders import load_bge_base_angle_emb

config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

llm = OpenAI(openai_api_base="http://localhost:1234/v1", openai_api_key="NULL", temperature=0.0)
embedding = load_bge_base_angle_emb()

loader = PyPDFLoader(file_path='../resources/test_rag_docs/test_rag.pdf')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

documents = loader.load_and_split(text_splitter=text_splitter)

persist_directory = 'test_rag_docs_pdf'
if not os.path.exists(persist_directory):
    vectordb = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

retriever = vectordb.as_retriever()

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)


def retrieve_content(question):
    response = qa({"question": question})
    print(question)
    return response["answer"]


llm_config = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0,
    "functions": [
        {
            "name": "retrieve_content",
            "description": "Answer any questions",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Use this function to answer any questions",
                    }
                },
                "required": ["question"],
            },
        }
    ],
}


def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    is_termination_msg=termination_msg,
    system_message="Reply TERMINATE in the end when everything is done.",
    function_map={"retrieve_content": retrieve_content}
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    llm_config=llm_config,
    is_termination_msg=termination_msg,
    max_consecutive_auto_reply=2,
    code_execution_config={"work_dir": "."},
    system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
    function_map={"retrieve_content": retrieve_content}
)

if __name__ == '__main__':
    # the assistant receives a message from the user, which contains the task description
    user_proxy.initiate_chat(
        assistant,
        message="""
    Find the answers to the questions:

    1. When Seraphina Celestia Moonshadow was born?

    Start the work now.
    """
    )
