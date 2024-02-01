from utils.loaders import load_dolphin

model = load_dolphin()

if __name__ == '__main__':
    model.invoke("What would be a good company name for a company that makes colorful socks?")
