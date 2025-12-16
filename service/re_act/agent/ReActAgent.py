# coding=utf-8
from service.re_act.config.AgentsLLMConfig import AgentsLLM
from service.re_act.execute.ToolExecutor import ToolExecutor
from service.re_act.agent.prompt.sys_prompt import SYS_PROMPT_TEMPLATE
import re


def _parse_output(text: str):
    """解析LLM的输出，提取Thought和Action。"""
    thought_match = re.search(r"Thought: (.*)", text)
    action_match = re.search(r"Action: (.*)", text)
    thought = thought_match.group(1).strip() if thought_match else None
    print(f"解析后完整thought: \n {thought} \n")
    action = action_match.group(1).strip() if action_match else None
    print(f"解析后完整action: \n {action} \n")
    return thought, action


def _parse_action(action_text: str):
    """解析Action字符串，提取工具名称和输入。"""
    match = re.match(r"(\w+)\[(.*)\]", action_text)
    if match:
        return match.group(1), match.group(2)
    return None, None


class ReActAgent:
    """
    循环执行, 推理使得行动更具目的性，而行动则为推理提供了事实依据
    Thought (思考) -> Action (行动) -> Observation (观察)
    """
    def __init__(self, llm_client: AgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = []  # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1.格式化提示词
            tools_desc = self.tool_executor.get_available_tools()
            history_str = "\n".join(self.history)
            react_prompt = SYS_PROMPT_TEMPLATE.get("REACT_PROMPT_TEMPLATE")
            prompt = react_prompt.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )
            print(f"prompt如下：\n {prompt} \n")

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # 3. 解析LLM的输出
            thought, action = _parse_output(response_text)

            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            tool_name, tool_input = _parse_action(action)
            if not tool_name or not tool_input:
                print("...无效的Action格式...")
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")

            tool_function = self.tool_executor.get_tool(tool_name)

            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input)  # 调用真实工具

            print(f"👀 观察: {observation}")

            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

            # 循环结束
        print("已达到最大步数，流程终止。")
        return None
