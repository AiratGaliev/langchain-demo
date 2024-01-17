from langchain.utilities import WikipediaAPIWrapper

wikipedia = WikipediaAPIWrapper()

if __name__ == '__main__':
    print(wikipedia.run('Langchain'))
