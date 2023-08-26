# 必要なライブラリとモジュールをインポート
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from urllib.parse import urlparse
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# langchain
from langchain import PromptTemplate, FewShotPromptTemplate, OpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import SystemMessage, HumanMessage

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

# 環境変数の読み込み
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = "gpt-3.5-turbo"  # モデル名を指定

# OpenAIのモデルを作成
llm = OpenAI(model_name=model_name, openai_api_key=openai_api_key)

# FewShotプロンプト準備 (開始)

## 教師データ
examples = [
    {
        "name": "Notion",
        "feature": "- オールインワークスペース: ノート、タスク、データベース、カレンダーなどを1つのプラットフォームで提供。\n- カスタマイズ可能: ページの自由なデザインや、必要なブロックの追加が可能。\n- 協力作業: チームメンバーとの共同編集やコメント、タスクの割り当て機能を持つ。",
        "examples": "- 知識ベースの作成: チームのナレッジやガイドラインを中央で管理。\n- タスク管理: プロジェクトのタスクや進捗をトラック。\n- 個人のノート取り: アイディアやリサーチのメモを整理。",
        "target_audience": "プロジェクトマネージャー、デザイナー、エンジニア、学生、教育者など",  # 対象者
        "tool_description": "Notionは、ノート、タスク、データベースなどの機能を統合したオールインワンの作業スペースを提供するツールです。"  # ツールの説明
    },
]

## 教師データのフォーマット
tool_formatter_template = """
## ツール名
{name}

## 特徴
{feature}

## 使用例
{examples}

## 対象者
{target_audience}

## ツール説明
{tool_description}
"""

## PromptTemplateのインスタンスを作成
tool_prompt_template = PromptTemplate(
    template=tool_formatter_template,
    input_variables=["name", "feature", "examples", "target_audience", "tool_description"]
)

## FewShotPromptTemplateのインスタンスを作成
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=tool_prompt_template,
    prefix="下記の出力形式を元に、",
    suffix="下記のツール紹介文も同様にマークダウン形式かつ日本語で出力してください {input}",
    input_variables=["input"],
    example_separator="\n\n",
)
# FewShotプロンプト準備 (終了)

# リクエストで送られてきたURLの解析

## URLを受け取るためのクラスを定義
class UrlQuery(BaseModel):
    url: str

## URLが有効かどうかをチェックする関数
def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

## 指定されたURLのコンテンツを取得する関数
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
      

# URLを受け取り、その内容を要約するAPIエンドポイントを定義
@app.post("/generate-article")
def summarize(query: UrlQuery):
    url = query.url
    is_valid_url = validate_url(url)

    if not is_valid_url:
        return {"error": "Please input valid url"}

    content = get_content(url)

    if content:
        prompt_text = few_shot_prompt.format(input=content)

        print("==================")
        print(prompt_text)
        print("==================")
        
        answer = llm.generate(prompts=[prompt_text])
        
        return {"generate-article": answer.generations[0][0].text.strip()}
    else:
        return {"error": "something went wrong"}