from utils.loaders import load_openhermes

model = load_openhermes()

if __name__ == '__main__':
    model.invoke("What would be a good company name for a company that makes colorful socks?")
