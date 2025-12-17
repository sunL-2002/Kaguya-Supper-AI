# coding=utf-8
from dotenv import load_dotenv
from armory.config.AgentsLLMConfig import AgentsLLM
from service.reflection.ReflectionAgent import ReflectionAgent


# --- Reflection 测试示例 ---
if __name__ == '__main__':
    # 1. 加载配置
    load_dotenv()

    # 2. 初始化连接
    llm_client = AgentsLLM()

    # 3. 初始化智能体
    agent = ReflectionAgent(llm_client)

    task = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
    # 4. 执行
    agent.run(task)


# 回答示例
"""
--- 开始处理任务 ---
任务: 编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。

--- 正在进行初始尝试 ---
🧠 正在调用 GLM-4.5-Flash 模型...
✅ 大语言模型响应成功:
```python
def find_primes(n: int) -> list[int]:
    ```
    Find all prime numbers between 1 and n (inclusive).

    This function uses the Sieve of Eratosthenes algorithm to efficiently
    identify prime numbers in the specified range.

    Parameters:
    n (int): The upper bound of the range (inclusive). Must be a positive integer.

    Returns:
    list[int]: A list of prime numbers from 1 to n. Returns an empty list if n < 2.

    Examples:
    >>> find_primes(10)
    [2, 3, 5, 7]
    >>> find_primes(1)
    []
    
    if n < 2:
        return []
    
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    return [i for i, prime in enumerate(is_prime) if prime]
```
📝 记忆已更新，新增一条 'execution' 记录。

--- 第 1/3 轮迭代 ---

-> 正在进行反思...
🧠 正在调用 GLM-4.5-Flash 模型...
✅ 大语言模型响应成功:
当前算法使用埃拉托斯特尼筛法，时间复杂度为O(n log log n)，这是理论最优的筛法复杂度。但存在以下可优化点：
1. **空间浪费**：数组包含所有整数（包括偶数），而偶数（除2外）均非素数，浪费50%空间。
2. **冗余标记**：内层循环标记所有倍数（包括偶数倍），而偶数倍已被2标记，造成冗余操作。

改进方案：**仅处理奇数的优化筛法**
- 空间优化：用数组仅存储奇数（索引i对应数2i+1），空间减半。
- 时间优化：外层循环仅遍历奇数，内层循环步长为2i（跳过偶数倍），减少标记次数。

改进后代码：
```python
def find_primes(n: int) -> list[int]:
    if n < 2:
        return []
    primes = [2]
    if n == 2:
        return primes
    size = (n + 1) // 2
    is_prime = [True] * size
    is_prime[0] = False  # 1 is not prime
    sqrt_n = int(n ** 0.5)
    for num in range(3, sqrt_n + 1, 2):
        idx = (num - 1) // 2
        if is_prime[idx]:
            start = num * num
            step = 2 * num
            for j in range(start, n + 1, step):
                is_prime[(j - 1) // 2] = False
    primes.extend(2 * i + 1 for i in range(1, size) if is_prime[i])
    return primes
```

优化效果：
- **空间**：从O(n)降至O(n/2)
- **时间**：内层循环次数减半（步长加倍），整体常数因子降低约50%
- **正确性**：通过索引映射确保奇数处理正确，结果与原始算法一致
📝 记忆已更新，新增一条 'reflection' 记录。

-> 正在进行优化...
🧠 正在调用 GLM-4.5-Flash 模型...
✅ 大语言模型响应成功:
```python
def find_primes(n: int) -> list[int]:
    
    Find all prime numbers between 1 and n (inclusive).

    This function uses an optimized Sieve of Eratosthenes algorithm that only
    processes odd numbers to reduce space usage and redundant operations.

    Parameters:
    n (int): The upper bound of the range (inclusive). Must be a positive integer.

    Returns:
    list[int]: A list of prime numbers from 1 to n. Returns an empty list if n < 2.

    Examples:
    >>> find_primes(10)
    [2, 3, 5, 7]
    >>> find_primes(1)
    []
    
    if n < 2:
        return []
    primes = [2]
    if n == 2:
        return primes
    size = (n + 1) // 2
    is_prime = [True] * size
    is_prime[0] = False  # 1 is not prime
    sqrt_n = int(n ** 0.5)
    for num in range(3, sqrt_n + 1, 2):
        idx = (num - 1) // 2
        if is_prime[idx]:
            start = num * num
            step = 2 * num
            for j in range(start, n + 1, step):
                is_prime[(j - 1) // 2] = False
    primes.extend(2 * i + 1 for i in range(1, size) if is_prime[i])
    return primes
```
📝 记忆已更新，新增一条 'execution' 记录。

--- 第 2/3 轮迭代 ---

-> 正在进行反思...
🧠 正在调用 GLM-4.5-Flash 模型...
✅ 大语言模型响应成功:

Process finished with exit code -1
"""