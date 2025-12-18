# coding=utf-8
from dotenv import load_dotenv
from armory.config.AgentsLLMConfig import AgentsLLM
from llm_test.way_of_thinking.plan_and_solve.PlanAndSolveAgent import PlanAndSolveAgent


# --- PlanAndSolve 测试示例 ---
if __name__ == '__main__':
    # 1. 加载配置
    load_dotenv()

    # 2. 初始化连接
    llm_client = AgentsLLM()

    # 3. 初始化智能体
    agent = PlanAndSolveAgent(llm_client)

    question = "一个工作岗位周一上了8h的班。周二工作的时间是周一的两倍。周三工作的时间比周二少了5h。请问这三天总共工作了多少工时？"
    # 4. 执行
    agent.run(question)


# ----- 执行结果 -----
"""
--- 开始处理问题 ---
问题: 一个工作岗位周一上了8h的班。周二工作的时间是周一的两倍。周三工作的时间比周二少了5h。请问这三天总共工作了多少工时？
---正在生成计划---
🧠 正在调用 GLM-4.5-Flash 模型...
✅ 大语言模型响应成功:
```python
["确定周一的工作时间", "计算周二的工作时间（基于周一的两倍）", "计算周三的工作时间（基于周二少5小时）", "计算三天的总工时（周一 + 周二 + 周三）"]
```
✅ 计划已生成:
```python
["确定周一的工作时间", "计算周二的工作时间（基于周一的两倍）", "计算周三的工作时间（基于周二少5小时）", "计算三天的总工时（周一 + 周二 + 周三）"]
```

--- 正在执行计划 ---

-> 正在执行步骤 1/4: 确定周一的工作时间
🧠 正在调用 GLM-4.5-Flash 模型...
✅ 大语言模型响应成功:
8
✅ 步骤 1 已完成，结果: 8

-> 正在执行步骤 2/4: 计算周二的工作时间（基于周一的两倍）
🧠 正在调用 GLM-4.5-Flash 模型...
✅ 大语言模型响应成功:
16
✅ 步骤 2 已完成，结果: 16

-> 正在执行步骤 3/4: 计算周三的工作时间（基于周二少5小时）
🧠 正在调用 GLM-4.5-Flash 模型...
✅ 大语言模型响应成功:
11
✅ 步骤 3 已完成，结果: 11

-> 正在执行步骤 4/4: 计算三天的总工时（周一 + 周二 + 周三）
🧠 正在调用 GLM-4.5-Flash 模型...
✅ 大语言模型响应成功:
35
✅ 步骤 4 已完成，结果: 35

--- 任务完成 ---
最终答案: 35

Process finished with exit code 0
"""