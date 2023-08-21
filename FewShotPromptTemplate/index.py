from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.llms import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# ツールの情報を定義
tools = [
    {
        "tool_name": "## ツール名\nChatGPT",
        "summary": "## 概要\nChatGPTは、OpenAIによって開発された自然言語処理（NLP）モデルで、人間のような会話ができる人工知能（AI）システムです。テキスト入力に対して、質問への回答を提供したり、対話を進めたりすることができます。ChatGPTは多岐にわたる用途で使用され、その柔軟性と高い生成能力により、さまざまな業界で支持されています。",
        "examples": "## 使用例\n- カスタマーサポートの自動化\n- 社内のFAQシステムの強化\n- 教育用の対話型学習ツール\n- 創作活動におけるアイデアの生成",
    },
    {
        "tool_name": "## ツール名\nNotion",
        "summary": "## 概要\nNotionは、プロジェクト管理、タスク管理、データベース作成、ノート取りなどを一つのプラットフォームで提供する統合されたコラボレーションツールです。個人、チーム、企業が効率的に作業できるように設計されています。",
        "examples": "## 使用例\n- プロジェクト管理のタスクボードの作成\n- 企業のナレッジベースの構築\n- 会議の議事録や計画書の作成\n- 個人的なジャーナルやブログの作成",
    },
    {
        "tool_name": "## ツール名\nCanva",
        "summary": "## 概要\nCanvaは、デザインを必要とするさまざまなプロジェクトに対応するオンラインのグラフィックデザインツールです。プロフェッショナルでなくても、ドラッグ&ドロップのインターフェイスと多数のテンプレートを使用して、高品質なグラフィックを作成することができます。",
        "examples": "## 使用例\n- ビジネスプレゼンテーションの作成\n- ソーシャルメディアの投稿画像の設計\n- イベントの招待状やポスターの制作\n- 個人的な写真の編集と共有",
    },
]


# プロンプトのテンプレートを定義（ツール名、概要、使用例をフォーマットするためのテンプレート）

tool_formatter_template = """
ツール名: {tool_name}
概要: {summary}
使用例: {examples}\n
"""

# PromptTemplateクラスを使ってプロンプトを作成
tool_prompt = PromptTemplate(
    template=tool_formatter_template,
    input_variables=["tool_name", "summary", "examples"],
)

few_shot_prompt = FewShotPromptTemplate(
    examples=tools, # モデルに学習する具体的な例を提供するリスト
    example_prompt=tool_prompt, # 与えられた例をフォーマットするためのPromptTemplateインスタンス
    prefix="ツールの詳細を教えて。", # プロンプトの冒頭に追加する固定のテキスト
    suffix="ツール名: {input}\n概要:", # 各入力例の後に追加されるテキスト
    input_variables=["input"], # 入力変数の名前のリスト
    example_separator="\n\n", # 例を区切るために使用される文字列
)

prompt_text = few_shot_prompt.format(input="VsCode")
print(prompt_text)

llm = OpenAI(openai_api_key=openai_api_key, model_name="text-davinci-003")
print(llm(prompt_text))
