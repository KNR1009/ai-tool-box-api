from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain import LLMChain

from fastapi import FastAPI
from langchain import PromptTemplate, OpenAI, LLMChain

app = FastAPI()  # FastAPIのインスタンスを作成


# OpenAIのモデルのインスタンスを作成
llm = OpenAI(model_name="text-davinci-003", openai_api_key="xxx")

# プロンプトのテンプレート文章を定義
template = """
次の料理のレシピを「## 概要」「## 材料」「## 手順」「## ワンポイント」でまとめてください
{sentences_before_check}
"""

# テンプレート文章にあるチェック対象の単語を変数化
prompt = PromptTemplate(
    input_variables=["sentences_before_check"],
    template=template,
)

# OpenAIのAPIにこのプロンプトを送信するためのチェーンを作成
chain = LLMChain(llm=llm, prompt=prompt,verbose=True)

# チェーンを実行し、結果を表示
print(chain("カレー"))