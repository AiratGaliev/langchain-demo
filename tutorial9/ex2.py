from langchain.utilities import PythonREPL

python_repl = PythonREPL()

if __name__ == '__main__':
    print(python_repl.run("print(17*2)"))
