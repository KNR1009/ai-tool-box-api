from langchain.document_loaders import SeleniumURLLoader

urls = [
    "https://udemy.benesse.co.jp/development/langchain.html",
]

loader = SeleniumURLLoader(urls=urls)

data = loader.load()

print("================")
print(data)
print("================")


