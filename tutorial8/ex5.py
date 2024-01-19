from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from pypdf import PdfReader

from utils.loaders import load_openhermes_llm, load_gte_base_emb

embedding = load_gte_base_emb()

llm = load_openhermes_llm()

doc_reader = PdfReader('../resources/impromptu-rh.pdf')

raw_text = ''
for i, page in enumerate(doc_reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

texts = text_splitter.split_text(raw_text)

docsearch = FAISS.from_texts(texts, embedding)

chain = load_qa_chain(llm,
                      chain_type="map_rerank",
                      return_intermediate_steps=True,
                      verbose=True
                      )

query = "who are openai?"
docs = docsearch.similarity_search(query, k=10)

if __name__ == '__main__':
    print(chain({"input_documents": docs, "question": query}, return_only_outputs=True))
