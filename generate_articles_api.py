from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import os
from dotenv import load_dotenv

# cros
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# .envよりファイルを取得
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = "gpt-3.5-turbo"  # または "gpt-3.5-turbo"
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name=model_name)


class UrlQuery(BaseModel):
    url: str


def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        if soup.main:
            return soup.main.get_text()
        elif soup.article:
            return soup.article.get_text()
        else:
            return soup.body.get_text()
    except:
        return None


def build_prompt(content, n_chars=1000):
    return f"""以下はとあるWebページのコンテンツである。内容を以下のような見出しでまとめてHTML形式で返してください。
下記はあくまでサンプル
<h2>特徴</h2>
<p>テキスト</p>
<h2>使用例</h2>
<p>テキスト</p>
<h2>対象者</h2>
<p>テキスト</p>
<h2>ツールの説明</h2>
<p>テキスト</p>
<h2>料金プラン</h2>
<p>テキスト</p>

========

{content[:n_chars]}

========

日本語で書いてね！
"""


@app.post("/generate-article")
def summarize(query: UrlQuery):
    url = query.url
    is_valid_url = validate_url(url)

    if not is_valid_url:
        return {"error": "Please input valid url"}

    content = get_content(url)
    if content:
        prompt = build_prompt(content)
        messages = [SystemMessage(content="You are a helpful assistant."), HumanMessage(content=prompt)]

        with get_openai_callback() as cb:
            answer = llm(messages)
        
        return {"generate-article": answer.content}
    else:
        return {"error": "something went wrong"}
