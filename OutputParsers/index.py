from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# CommaSeparatedListOutputParserのインスタンスを作成し、カンマで区切られたリストを解析するためのパーサーを定義します。
output_parser = CommaSeparatedListOutputParser()

# パーサーから形式指示子を取得し、プロンプトの一部として使用します。
format_instructions = output_parser.get_format_instructions()

# PromptTemplateのインスタンスを作成し、プロンプトのテンプレートを定義します。このテンプレートは、指定された主題に関連する5つの項目をリストします。
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

# OpenAIのインスタンスを作成し、指定されたAPIキーとモデル名でLLM（Language Model）を初期化します。
llm = OpenAI(openai_api_key=openai_api_key, model_name="text-davinci-003")

# プロンプトをフォーマットし、指定された主題（この場合は「Programming Language」）で入力を作成します。
_input = prompt.format(subject="Programming Language")

# LLMに入力を渡し、出力を取得します。
output = llm(_input)

# パーサーを使用して出力を解析し、結果を表示します。
print(output_parser.parse(output))
