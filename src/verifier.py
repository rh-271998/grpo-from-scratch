import re
from collections import Counter
import ast
from transformers import AutoTokenizer
import math
def check_answer_validity(ansText):

    allowed = [' ', '(', ')', '-', '+', '*', '/']

    for s in ansText:
        
        if (s not in allowed) and (ord(s) < 48 or ord(s) > 57):
            return False
    
    return True

def check_multiset(ansText, nums):

    found_ints = [int(num) for num in re.findall(r"\d+", ansText)]
    return Counter(found_ints) == Counter(nums)

def evaluate(node):

    if isinstance(node, ast.Expression):
        return evaluate(node.body)
    
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("disallowed node")

    if isinstance(node, ast.BinOp):
        left = evaluate(node.left)
        right = evaluate(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            if right != 0:
                return left / right
            else:
                raise ZeroDivisionError()
        else:
            raise ValueError("disallowed node")
    
    if isinstance(node, ast.UnaryOp):
        operand = evaluate(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return operand
        else:
            raise ValueError("disallowed node")
    raise ValueError("disallowed node")



def compute_reward(response, nums, target):

    reward = 0.0

    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    thinkContent = re.search(r'<think>(.*?)</think>', response, re.DOTALL)

    if match:
        reward += 0.1
        ansContent = match.group(1).strip()

        if len(ansContent) > 0:
            allowedChars = check_answer_validity(ansContent)
            if allowedChars:
                multisetCheck = check_multiset(ansContent, nums)
                if multisetCheck:
                    try:
                        tree = ast.parse(ansContent, mode='eval')
                        value = evaluate(tree)

                        if abs(value - target) < 1e-6:
                            reward += 1.0
                        else:
                            reward += 0.5 * math.exp(-abs(value - target) / 5.0)

                    except Exception:
                        pass


    if thinkContent:
        reward += 0.1
    
    return reward


def build_prompt(nums, target, tokenizer):

    messages = [{"role": "system",
                 "content":"You solve Countdown problems. Using each given number exactly once and the "
                            "operations +, -, *, / (parentheses allowed), build an arithmetic expression "
                            "that equals the target. Put your reasoning inside <think> </think> and the "
                            "final expression inside <answer> </answer>. Think step by step."
                            "Format: <think> ... </think><answer> ... </answer>"},
                 {"role": "user",
                 "content": f"Numbers: {nums}, target: {target}. Find an expression equal to the target."}]

    template = tokenizer.apply_chat_template(messages,
                                             tokenize=False,
                                             add_generation_prompt=True)
    return template

if __name__=="__main__":
    modelName = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(modelName)

    print(build_prompt([1,2,3,4], 10, tokenizer))
