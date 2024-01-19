from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from pypdf import PdfReader

from utils.loaders import load_gte_base_emb

embedding = load_gte_base_emb()

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

query = "how does GPT-4 change social media?"
docs = docsearch.similarity_search(query)

if __name__ == '__main__':
    print(len(docs))
    print(docs)
