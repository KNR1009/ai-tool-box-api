# 必要なライブラリとモジュールをインポート
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

# CORS対応のためのミドルウェアをインポート
from fastapi.middleware.cors import CORSMiddleware

# FastAPIのインスタンスを作成
app = FastAPI()

# CORSの設定を追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# .envファイルから環境変数を読み込む
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = "gpt-3.5-turbo"  # モデル名を指定
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name=model_name)

# URLを受け取るためのクラスを定義
class UrlQuery(BaseModel):
    url: str

# URLが有効かどうかをチェックする関数
def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# 指定されたURLのコンテンツを取得する関数
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

# ChatGPTに送信するためのプロンプトを作成する関数
def build_prompt(content, n_chars=300):
    return f"""以下はとある。Webページのコンテンツである。内容を「## ツール名」「## 特徴」「## 使用例」「## 対象者(箇条書き)」「## ツールの説明」の見出しでそれぞれまとめてください

========

{content[:1000]}

========

日本語で書いてね！
"""


# URLを受け取り、その内容を要約するAPIエンドポイントを定義
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
