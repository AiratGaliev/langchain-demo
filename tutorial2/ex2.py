from langchain_experimental.pal_chain import PALChain
from langchain_experimental.utilities import PythonREPL

from utils.loader import load_openhermes

llm = load_openhermes()

pal_chain = PALChain.from_math_prompt(llm, verbose=True)

question_01 = "Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy. If Cindy has four pets, how many total pets do the three have?"

question_02 = "The cafeteria had 23 apples. If they used 20 for lunch and bought 6 more, how many apples do they have?"

python_repl = PythonREPL()

if __name__ == '__main__':
    result = pal_chain.run(question_01)
    print(result)
    result = pal_chain.run(question_02)
    print(result)
