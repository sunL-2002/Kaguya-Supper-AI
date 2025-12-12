# coding=utf-8
import os
from operator import itemgetter
from langchain_community.chat_models import ChatZhipuAI
from dotenv import load_dotenv
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
import tiktoken

""" 聊天机器人 """

# 加载.env配置
load_dotenv()

# 2. 模型
model = ChatZhipuAI(
    model_name="GLM-4.5-Flash",
    temperature=0.8,
    api_key=os.getenv("ZHIPUAI_API_KEY")
)

# 记忆存储
store = {}
# 直接使用 tiktoken 的编码器
encoder = tiktoken.get_encoding("cl100k_base")


# 自定义的token解析器
def my_token_counter(text) -> int:
    if not isinstance(text, list):
        text = [text]
    return sum(len(encoder.encode(msg.content)) for msg in text)


# 修剪器，控制保存的令牌数量,, TiktokenCounter 分词器，计算tokens
trimmer = trim_messages(
    max_tokens=10,
    strategy="last",
    token_counter=my_token_counter,
    include_system=True,
    allow_partial=False,
    start_on="human",
)


# 记忆存储方法
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 提示词模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
# 构建链式调用， 先进行分词计算token
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

with_message_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")

config = {"configurable": {"session_id": "abc2"}}


response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="whats my name?")],
        "language": "English"
    },
    config=config,
)
print(response.content)

response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="what math problem did i ask?")],
        "language": "English",
    },
    config=config,
)
print(response.content)

# 流式调用
config1 = {"configurable": {"session_id": "abc15"}}
for r in with_message_history.stream(
    {
        "messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
        "language": "English",
    },
    config=config1,
):
    print(r.content, end="|")
