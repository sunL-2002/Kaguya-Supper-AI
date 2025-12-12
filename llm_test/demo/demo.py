# coding=utf-8
import os
from langchain_community.chat_models import ChatZhipuAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 从.env文件加载环境变量,默认加载项目根目录下的.env文件
load_dotenv()


# 初始化智谱AI模型
zhipu_llm = ChatZhipuAI(
    model_name="GLM-4.5-Flash",  # 使用智谱最新的GLM-4模型[1,4](@ref)
    temperature=0.8,  # 适当提高创造性[4](@ref)
    api_key=os.getenv("ZHIPUAI_API_KEY")  # 从环境变量读取API密钥[2,5](@ref)
)

# 输出解析器
parser = StrOutputParser()


def langchain_call1():
    messages = [
        SystemMessage(content="Translate the following from English into Italian"),
        HumanMessage(content="hi!"),
    ]

    # 链式调用
    chain = zhipu_llm | parser
    result = chain.invoke(messages)

    print(result)


def tip_temp():
    """ 使用提示词模板 """
    system_template = "Translate the following into {language}:"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )

    result = prompt_template.invoke({"language": "italian", "text": "hi"})

    # 返回一个 ChatPromptValue，由两个消息组成
    # [SystemMessage(content='Translate the following into italian:'),
    #  HumanMessage(content='hi')]
    result.to_messages()

    chain = prompt_template | zhipu_llm | parser
    chain.invoke({"language": "italian", "text": "hi"})


if __name__ == '__main__':
    langchain_call1()
