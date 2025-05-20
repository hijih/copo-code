# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from verl.utils.reward_score.math import *
from collections import defaultdict
try:
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

import re
import signal
from typing import Optional


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.
    
    Args:
        string: Input string containing LaTeX code
        
    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return string

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx:right_brace_idx + 1] if right_brace_idx is not None else string


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string if present.

    Args:
        s: String, maybe with format "\\boxed{content}"

    Returns:
        The content inside the boxed command, or original string if not boxed.
    """
    if not isinstance(s, str):  # 处理 None 或其他非字符串情况
        return s
    left = "\\boxed{"
    right = "}"
    if s.startswith(left) and s.endswith(right):
        return s[len(left):-1]
    return s  # 如果不是 boxed 格式，直接返回原始字符串


def extract_solution(solution_str):
    solution_str = remove_boxed(last_boxed_only_string(solution_str))
    for before, after in SUBSTITUTIONS:
        final_answer = solution_str.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    return final_answer

class timeout:

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.
    
    Args:
        final_answer: The answer string to normalize
        
    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(solution_str: str,
                       gt: str,
                       gt_need_extract: bool = False,
                       answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)") -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.
    
    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer
        
    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(pred: str,
                          gt: str,
                          pause_tokens_index: Optional[list[int]] = None) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.
    
    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens
        
    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100:]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(solution_str: str,
           answer: str,
           strict_box_verify: bool = False,
           pause_tokens_index: Optional[list[int]] = None) -> bool:
    """Verify if the solution is correct.
    
    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens
        
    Returns:
        True if the solution is correct, False otherwise
    """
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred

def is_true(model_output: str, ground_truth: str) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.
    
    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    # except Exception as e:
    except:
        if model_output == ground_truth:
            ret_score = 1.0
    if ret_score == 0.0:
        if model_output == ground_truth:
            ret_score = 1.0
    return ret_score

def preprocess_latex(expr_str: str) -> str:
    """将 LaTeX 表达式转换为更容易 sympy 解析的格式"""

    expr_str = expr_str.strip()

    # 0. 去除包裹在 $...$ 或 $$...$$ 中的内容
    expr_str = re.sub(r'^\s*\${1,2}(.*?)\${1,2}\s*$', r'\1', expr_str)

    # 1. 处理 \boxed{...} → ...
    expr_str = re.sub(r'\\boxed{([^{}]+)}', r'\1', expr_str)

    # 2. 处理 \text{...} → ...
    expr_str = re.sub(r'\\text{([^{}]+)}', r'\1', expr_str)

    # 3. 去掉 \left 和 \right
    expr_str = expr_str.replace(r'\left', '')
    expr_str = expr_str.replace(r'\right', '')

    # 4. 替换 \frac{a}{b} → (a)/(b)
    expr_str = re.sub(r'\\frac{([^{}]+)}{([^{}]+)}', r'(\1)/(\2)', expr_str)

    # 5. 替换 \cdot, \times → *
    expr_str = expr_str.replace(r'\cdot', '*')
    expr_str = expr_str.replace(r'\times', '*')

    # 6. 处理 \\ → \ （避免双反斜杠转义失败）
    expr_str = expr_str.replace('\\\\', '\\')

    # 7. 删除多余空格
    expr_str = expr_str.strip()
    expr_str = expr_str.replace('$', '')
    return expr_str

def tag_count_reward(solution_str) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    reward = 0.0
    
    think_match = re.search(r"<think>(.*?)</think>", solution_str)
    think = bool(think_match) and len(think_match.group(1).strip()) == 1
    
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str)
    answer = bool(answer_match) and len(answer_match.group(1).strip()) == 1
    
    if think and answer:
        reward = 0.25
        
    return reward

def extract_last_answer_block(text):
    # 找到所有以 'Answer:' 开头的位置
    matches = list(re.finditer(r'Answer:', text))
    if not matches:
        return ''

    start = matches[-1].end()
    end = len(text)
    return text[start:end].strip()

def accuracy_reward(data_source, solution_str, ground_truth, extra_info=None):
    """Reward function that checks if the completion is the same as the ground truth."""

    reward = 0.0
    ground_truth = ground_truth.lower().strip()
    ground_truth = ground_truth.replace('\n', ' ')

    origin_solution = solution_str
    solution_str = preprocess_latex(solution_str)
    answer = extract_last_answer_block(solution_str)
    origin_answer = answer
    answer = preprocess_latex(answer)

    if answer is None:
        answer = origin_answer

    if answer == '':
        answer =  extract_solution(origin_solution)
        if answer is None:
            answer = ''
            reward = 0.0
            return reward, answer, answer
    
    answer = answer.lower().strip()

    if is_true(answer, ground_truth):
        reward = 1.0
    else:
        if ground_truth == 'true':  
            ground_truth = '1'
        elif ground_truth == 'false':
            ground_truth = '0'
        if answer == 'true':  
            answer = '1'
        elif answer == 'false':
            answer = '0'
        answer = answer.replace('without the quotes', '')
        answer = answer.lower().strip()
        answer = answer.replace('\\', '')
        answer = answer.replace('(', '')
        answer = answer.replace(')', '')
        if is_true(answer, ground_truth):
            reward = 1.0
        else:
            match = re.search(r"\(([A-Da-d])\)", answer)
            if match:
                answer = match.group(0)
                if is_true(answer, ground_truth):
                    reward = 1.0
                
    # print(f"answer: {answer}, gt: {ground_truth}, reward: {reward}")
    return reward, answer, answer
    # strict_box_verify = False
    # correct, pred = verify(solution_str, ground_truth, strict_box_verify)
    # if correct:
    #     reward = 1.0

    # return reward, pred, pred

def compute_score(data_source, solution_str: str, ground_truth: str, extra_info=None) -> float:
    
    results = {}
    results['score'], results['extract_answer'], results['pred'] = accuracy_reward(data_source, solution_str, ground_truth)
    return results
