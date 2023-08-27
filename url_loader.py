from langchain.document_loaders import SeleniumURLLoader

urls = [
    "https://aisaas.pkshatech.com/chatbot",
]

loader = SeleniumURLLoader(urls=urls)

data = loader.load()

print("================")
print(data)
print("================")


