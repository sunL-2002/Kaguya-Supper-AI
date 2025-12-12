# coding=utf-8
from zhipuai import ZhipuAI


class ChatGLMClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.client = ZhipuAI(api_key=api_key)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应"""
        print("正在调用大语言模型...")
        try:
            message = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                temperature=0.7,
                max_tokens=1024,
                stream=False
            )
            print("*****************")
            print(f"大模型响应: {response}")
            answer = response.choices[0].message.reasoning_content
            print("大语言模型响应成功。")
            print("*****************")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"
