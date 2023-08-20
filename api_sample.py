from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from pydantic import BaseModel

import os
from dotenv import load_dotenv

app = FastAPI()

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数からOpenAIのAPIキーを取得する
openai_api_key = os.getenv("OPENAI_API_KEY")

class Query(BaseModel):
    question: str

# インスタンスの生成や設定は一度だけ行うため、ここに配置
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

@app.post("/api/chat")
def get_answer(query: Query):
    # ユーザーからの質問をHumanMessageとして扱う
    user_message = HumanMessage(content=query.question)
    messages = [user_message]

    # AIからの応答を取得
    response = llm(messages)

    # AIMessageとしての応答を返す
    answer = AIMessage(content=response.content)
    return answer
