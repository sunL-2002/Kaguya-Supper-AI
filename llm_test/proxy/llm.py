# coding=utf-8
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
import requests
import json
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class BaiduMCPSearchInput(BaseModel):
    query: str = Field(description="搜索查询内容")
    stream: bool = Field(default=False, description="是否使用流式响应")


class BaiduMCPSearchTool(BaseTool):
    name: str = "baidu_mcp_search"
    description: str = "使用百度AI搜索MCP服务器进行实时信息搜索"
    args_schema: type[BaseModel] = BaiduMCPSearchInput

    def __init__(self, bearer_token: str):
        super().__init__()
        self._base_url = "https://qianfan.baidubce.com/v2/ai_search/mcp"
        self._headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        }

    def _run(self, query: str, stream: bool = False) -> str:
        payload = {
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {
                    "query": query,
                    "stream": stream
                }
            }
        }

        try:
            response = requests.post(
                self._base_url,
                headers=self._headers,
                json=payload,
                timeout=30  # 添加超时
            )

            if response.status_code == 200:
                data = response.json()
                # 根据实际API响应结构调整
                if "result" in data:
                    return data["result"]
                elif "data" in data:
                    return str(data["data"])
                else:
                    return f"搜索完成，响应: {str(data)[:200]}..."
            else:
                return f"搜索失败，状态码: {response.status_code}"

        except Exception as e:
            return f"搜索出错: {str(e)}"

    async def _arun(self, query: str, stream: bool = False) -> str:
        # 异步版本
        import aiohttp

        payload = {
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {
                    "query": query,
                    "stream": stream
                }
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self._base_url,
                        headers=self._headers,
                        json=payload,
                        timeout=30
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("result", "搜索完成")
                    else:
                        return f"搜索失败，状态码: {response.status}"
        except Exception as e:
            return f"异步搜索出错: {str(e)}"


# 主程序
load_dotenv()


def main():
    # 创建记忆存储
    memory = MemorySaver()

    # 初始化模型
    model = ChatZhipuAI(
        model_name="GLM-4.5-Flash",
        temperature=0.8,
        api_key=os.getenv("ZHIPUAI_API_KEY")
    )

    # 创建搜索工具
    search_tool = BaiduMCPSearchTool(bearer_token=os.getenv("BAIDU_MCP_TOKEN"))
    tools = [search_tool]

    # 创建agent
    agent_executor = create_react_agent(model, tools, checkpointer=memory)

    # 配置会话ID
    config = {"configurable": {"thread_id": "abc123"}}

    print("=== 智能助手已启动 ===")
    print("输入 'quit' 退出")
    print("输入 'clear' 清除记忆")
    print("-" * 30)

    # 交互循环
    while True:
        user_input = input("\n你: ").strip()

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            # 清除记忆，创建新的thread_id
            config = {"configurable": {"thread_id": f"thread_{len(config)}"}}
            print("记忆已清除")
            continue
        elif not user_input:
            continue

        try:
            print("\n助手: ", end="", flush=True)

            # 流式响应
            for chunk in agent_executor.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config
            ):
                if 'agent' in chunk:
                    # Agent的思考过程
                    if 'messages' in chunk['agent']:
                        for msg in chunk['agent']['messages']:
                            if hasattr(msg, 'content') and msg.content:
                                print(msg.content, end="", flush=True)

                elif 'tools' in chunk:
                    # 工具调用结果
                    if 'messages' in chunk['tools']:
                        for msg in chunk['tools']['messages']:
                            if hasattr(msg, 'content') and msg.content:
                                print(f"[搜索结果: {msg.content[:100]}...]", end="", flush=True)

            print()  # 换行

        except Exception as e:
            print(f"\n出错: {str(e)}")


if __name__ == "__main__":
    main()
