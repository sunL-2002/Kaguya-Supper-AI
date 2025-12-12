# coding=utf-8
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
import os
from langchain_community.chat_models import ChatZhipuAI
from dotenv import load_dotenv

load_dotenv()
"""
翻译对话模型  translate-llm_test
"""

# 1. 提示词模板
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. 模型
model = ChatZhipuAI(
    model_name="GLM-4.5-Flash",
    temperature=0.8,
    api_key=os.getenv("ZHIPUAI_API_KEY")
)

# 3. 文本解析器
parser = StrOutputParser()

# 4. 链式调用
chain = prompt_template | model | parser

# 5. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
