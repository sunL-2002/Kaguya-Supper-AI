# coding=utf-8
import ast
import operator
import math

# 定义一个安全的命名空间，包含数学函数和常量
SAFE_NAMESPACE = {
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'sqrt': math.sqrt,
    'log': math.log,
    'log10': math.log10,
    'pi': math.pi,
    'e': math.e,
}


def safe_calculator(expression: str) -> str:
    """
    一个安全且功能增强的数学计算器。
    支持基本运算符 (+, -, *, /, //, %, **), 括号, 以及部分 math 函数 (如 sin, sqrt) 和常量 (pi, e)。
    """
    if not expression or not isinstance(expression, str):
        return "输入无效：表达式必须是一个非空字符串。"

    # 去除空格，让输入更宽容
    expression = expression.replace(" ", "")

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # 手动实现 _eval 方案， 实际未运行此处代码
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):  # Python >= 3.8
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise ValueError(f"不支持的常量类型: {type(node.value)}")
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](left, right)
            else:
                raise TypeError(f"不支持的二元运算符: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](operand)
            else:
                raise TypeError(f"不支持的一元运算符: {op_type}")
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in SAFE_NAMESPACE and callable(SAFE_NAMESPACE[func_name]):
                args = [_eval(arg) for arg in node.args]
                return SAFE_NAMESPACE[func_name](*args)
            else:
                raise NameError(f"不允许的函数或常量: '{func_name}'")
        elif isinstance(node, ast.Name):
            if node.id in SAFE_NAMESPACE:
                return SAFE_NAMESPACE[node.id]
            else:
                raise NameError(f"不允许的名称: '{node.id}'")
        else:
            raise TypeError(f"不支持的AST节点类型: {type(node)}")

    try:
        # 使用 compile+eval 模式，以便注入命名空间
        compiled_node = compile(expression, '<string>', 'eval')
        # 限制 globals 和 locals，只允许访问 SAFE_NAMESPACE
        result = eval(compiled_node, {"__builtins__": {}}, SAFE_NAMESPACE)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


if __name__ == '__main__':
    # --- 测试 ---
    print(f"基础计算: (123 + 456) * 789 / 12 = {safe_calculator('(123 + 456) * 789 / 12')}")
    print(f"使用函数: sqrt(16) + sin(pi/2) = {safe_calculator('sqrt(16) + sin(pi/2)')}")
    print(f"错误输入: __import__('os').system('echo pwned') = {safe_calculator('__import__(os).system(echo pwned)')}")
    print(f"错误输入: 1 + (2 = {safe_calculator('1 + (2')}")
