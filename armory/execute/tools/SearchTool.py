# coding=utf-8
from tavily import TavilyClient
import os


def search(query: str) -> str:
    """
    一个基于Tavily Search API的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [Tavily Search API] 网页搜索: {query}")
    try:
        # 1. 从环境变量中读取API密钥
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"

        # 2. 初始化Tavily客户端
        tavily = TavilyClient(api_key=api_key)

        # 3. 构造查询参数
        params = {
            "query": query,
            "search_depth": "basic",
            "include_answer": True
        }

        # 4. 调用API，include_answer=True会返回一个综合性的回答
        response = tavily.search(**params)
        # print(f"原始回答：\n {response} \n")

        # 5. Tavily返回的结果已经非常干净，可以直接使用
        # response['answer'] 是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]

            # 如果没有综合性回答，则格式化原始结果
            formatted_results = []
            for result in response.get("results", []):
                formatted_results.append(f"- {result['title']}: {result['content']}")

            if not formatted_results:
                return f"抱歉，没有找到关于 '{query}' 的信息。"

            return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"
