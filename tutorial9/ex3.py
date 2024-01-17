from langchain.utilities import DuckDuckGoSearchAPIWrapper

search = DuckDuckGoSearchAPIWrapper()

if __name__ == '__main__':
    print(search.run("Tesla stock price?"))
