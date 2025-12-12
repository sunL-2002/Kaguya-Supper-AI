# coding=utf-8
# 导入所需库
from typing import Optional

from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import Annotated, TypedDict


# 从.env文件加载环境变量
load_dotenv()  # 默认加载项目根目录下的.env文件[6,7,8](@ref)

# 初始化智谱AI模型
zhipu_llm = ChatZhipuAI(
    model_name="GLM-4.5-Flash",  # 使用智谱最新的GLM-4模型[1,4](@ref)
    temperature=0.8,  # 适当提高创造性[4](@ref)
    api_key=os.getenv("ZHIPUAI_API_KEY")  # 从环境变量读取API密钥[2,5](@ref)
)


class Joke(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]


def langchain_call1():
    # 定义童话故事生成提示模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一位著名的童话故事大王，擅长创作富有教育意义的奇幻故事。"),
        ("human", """
    请创作一个短篇童话故事，要求：
    1. 主题：永远保持一颗善良的心
    2. 目标读者：6-12岁儿童
    3. 包含魔法或奇幻元素
    4. 故事要生动有趣，激发儿童的想象力和学习兴趣
    5. 通过故事自然地传达善良的价值和意义
    6. 故事长度约200字
    请用以下格式创作：
    ---
    标题：《有创意的标题[](@replace=10001)》
    正文：（分3-5个自然段，每段有明确的情节发展）
    故事寓意：（用1-2句话总结故事传达的道理）
    ---
    """)
    ])

    json_schema = {
        "title": "joke",
        "description": "Joke to tell user.",
        "type": "object",
        "properties": {
            "setup": {
                "type": "string",
                "description": "The setup of the joke",
            },
            "punchline": {
                "type": "string",
                "description": "The punchline to the joke",
            },
            "rating": {
                "type": "integer",
                "description": "How funny the joke is, from 1 to 10",
                "default": None,
            },
        },
        "required": ["setup", "punchline"],
    }
    # 创建处理链
    # story_chain = prompt_template | zhipu_llm | StrOutputParser()
    llm = zhipu_llm.with_structured_output(json_schema, include_raw=True)
    # 生成童话故事
    try:
        fairy_tale = llm.invoke("Tell me a joke about cats")
        print("✨ 生成的童话故事 ✨")
        print(fairy_tale)
    except Exception as e:
        print(f"生成故事时出错: {str(e)}")


if __name__ == '__main__':
    langchain_call1()
