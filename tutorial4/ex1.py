# https://colab.research.google.com/drive/13P3DLQEfjG1xjLKd9UT9-IozsDomXrFJ#scrollTo=VkVTT54xNq8T
from langchain_experimental.pal_chain import PALChain

from utils.loader import load_openhermes

llm = load_openhermes()

pal_chain = PALChain.from_math_prompt(llm, verbose=True)

question_01 = "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
question_02 = "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"
question_03 = "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"
question_04 = "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"
question_05 = "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"
question_06 = "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"
question_07 = "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"
question_08 = "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"

if __name__ == '__main__':
    result = pal_chain.run(question_01)
    print(result)
    result = pal_chain.run(question_02)
    print(result)
    result = pal_chain.run(question_03)
    print(result)
    result = pal_chain.run(question_04)
    print(result)
    result = pal_chain.run(question_05)
    print(result)
    result = pal_chain.run(question_06)
    print(result)
    result = pal_chain.run(question_07)
    print(result)
    result = pal_chain.run(question_08)
    print(result)
