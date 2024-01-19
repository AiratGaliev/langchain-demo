from utils.loaders import load_openhermes

llm = load_openhermes()

if __name__ == '__main__':
    llm.invoke("Why did the chicken cross the road?")
