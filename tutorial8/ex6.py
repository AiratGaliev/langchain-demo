from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from pypdf import PdfReader

from utils.loader import load_openhermes_llm, load_gte_base_emb

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

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 4})

rqa = RetrievalQA.from_chain_type(llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

if __name__ == '__main__':
    rqa("What does gpt-4 mean for creativity?")
    print("\n")
    rqa("What have the last 20 years been like for American journalism?")
    print("\n")
    rqa("How can journalists use GPT-4?")
    print("\n")
    rqa("How is GPT-4 different from other models?")
    print("\n")
    rqa("What is beagle Bard?")
