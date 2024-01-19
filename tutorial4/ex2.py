# https://colab.research.google.com/drive/13P3DLQEfjG1xjLKd9UT9-IozsDomXrFJ#scrollTo=cs8b3-DW_7Lc
from langchain_experimental.pal_chain import PALChain

from utils.loaders import load_openhermes

llm = load_openhermes()

pal_chain = PALChain.from_math_prompt(llm, verbose=True)


def check_result(answer: str, result):
    if answer in str(result).lower():
        print(True)
    else:
        print(False)


question_01 = "John buys twice as many red ties as blue ties. The red ties cost 50% more than blue ties. He spent $200 on blue ties that cost $40 each. How much did he spend on ties?"
question_02 = "Maggie spent a quarter of her money, while Riza spent one-third of her money. They each had $60. How much money do the two of them have left?"
question_03 = "In November, a toy was $8186321.4112. In December, the price increased by 80%. In January, the price decreased by 50%. What was the price of the toy after it was discounted in January?"
question_04 = "Repeat cheese seven times; every third say whiz"
question_05 = "Say the letters of the alphabet in capital letters, but only the odd ones"
question_06 = "If you have 80 tickets for the fair and each ride costs 5 tickets, how many rides can you go on?"
question_07 = "The school has $20,000 to buy new computer equipment. If each piece of equipment costs $50, how many pieces can the school buy in total?"
question_08 = "An Italian restaurant receives a shipment of 86 veal cutlets. If it takes 3 cutlets to make a dish, how many cutlets will the restaurant have left over after making as many dishes as possible?"
question_09 = "There are 235 books in a library. On Monday, 123 books are taken out. On Tuesday, 56 books are brought back. How many books are there now?"
question_10 = "The school’s junior band has 10 saxophone players and 20 trumpet players.  The school’s senior band has 18 saxophone players and 29 trumpet players. Which band has the higher ratio of trumpet to saxophone players?"
question_11 = "If you wake up at 7:00 a.m. and it takes you 1 hour and 30 minutes to get ready and walk to school, at what time will you get to school?"
question_12 = "The ratio of Jenny’s trophies to Meredith’s trophies is 7:4. The difference between the numbers is 12. What are the numbers?"

if __name__ == '__main__':
    result = pal_chain.run(question_01)
    print(result)
    check_result("800.0", result)
    result = pal_chain.run(question_02)
    print(result)
    check_result("85.0", result)
    result = pal_chain.run(question_03)
    print(result)
    check_result("7367689.27008", result)
    result = pal_chain.run(question_04)
    print(result)
    check_result("cheese cheese cheese whiz cheese cheese cheese whiz cheese", result)
    result = pal_chain.run(question_05)
    print(result)
    check_result("ACEGIKMOQSUWY", result)
    result = pal_chain.run(question_06)
    print(result)
    check_result("16", result)
    result = pal_chain.run(question_07)
    print(result)
    check_result("400", result)
    result = pal_chain.run(question_08)
    print(result)
    check_result("2", result)
    result = pal_chain.run(question_09)
    print(result)
    check_result("168", result)
    result = pal_chain.run(question_10)
    print(result)
    check_result("junior", result)
    result = pal_chain.run(question_11)
    print(result)
    check_result("8:30", result)
    result = pal_chain.run(question_12)
    print(result)
    check_result("[28.0, 16.0]", result)
