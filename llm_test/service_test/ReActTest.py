# coding=utf-8
from armory.execute.tools.SearchTool import search
from dotenv import load_dotenv
from armory.config.AgentsLLMConfig import AgentsLLM
from armory.execute.ToolExecutor import ToolExecutor
from service.re_act.agent.ReActAgent import ReActAgent

# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    load_dotenv()

    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.register_tool("Search", search_description, search)

    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.get_available_tools())

    # 4. 智能体的Action调用，问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.get_tool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")


# --- 客户端使用示例 ---
if __name__ == '__main__':
    # 0. 加载.env配置
    load_dotenv()

    try:
        # 1. 构建LLM客户端
        llmClient = AgentsLLM()

        # 2. 构建消息体
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        # 3. 调用LLM问答测试
        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)


# --- ReActAgent 使用示例 ---
if __name__ == '__main__':
    # 0. 加载.env配置
    load_dotenv()

    # 1. 构建LLM客户端
    llmClient = AgentsLLM()

    # 2. 构建工具执行器
    toolExecutor = ToolExecutor()
    # 2.1 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.register_tool("Search", search_description, search)

    # 3. 构建用户消息
    question = "华为最新手机型号及主要卖点"

    # 4. 创建agent
    agent = ReActAgent(llmClient, toolExecutor)

    # 5. 执行
    agent.run(question)
