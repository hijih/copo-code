# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""
import re
import os 
from collections import defaultdict, Counter
from sympy import sympify, simplify
from sympy.parsing.latex import parse_latex
import numpy as np
import torch
import json
import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns

def tensor_to_list(obj):
    if torch.is_tensor(obj):
        return obj.tolist()
    elif isinstance(obj, list):
        return [tensor_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    else:
        return obj
    
def compute_score_global(local_reward: list[torch.Tensor], extract_answer, reward_coef_flg, output_file) -> float:
    roll_n = len(local_reward)
    lamda = (1- 1 / roll_n) / (np.log2(roll_n))
    
    extract_answer_clean = tensor_to_list(extract_answer)
    local_reward_clean = tensor_to_list(local_reward)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "a") as f:
        json_line = json.dumps({
            "answer_list": extract_answer_clean,
            "reward_list": local_reward_clean
        })
        f.write(json_line + "\n")

    def compute_entropy(prob_dist: torch.Tensor, eps: float = 1e-8):
        # 确保是 tensor
        prob_dist = torch.tensor(prob_dist, dtype=torch.float32)

        # 屏蔽掉 <= 0 的概率（避免 log 出错）
        prob_dist = prob_dist[prob_dist > 0]

        # 防止 log(0) 出错，用 eps 稳定
        entropy = -torch.sum(prob_dist * torch.log2(prob_dist + eps))
        return entropy

    def entropy_from_list(data):
        counts = Counter(data)
        total = sum(counts.values())
        probs = [count / total for count in counts.values()]
        return compute_entropy(probs)

    def canonicalize_expressions(expr_list):
        """
        将表达式列表中等价的数学表达式归一化表示，支持 LaTeX 和普通格式。
        返回和输入等长的 list，等价表达式会统一为相同的 canonical string。
        """
        simplified_map = {}
        result = []

        def preprocess_latex(expr_str):
            """预处理 LaTeX 表达式，替换 '\\' 和 '\\frac' 之类的符号"""
            # cjw:添加了对$符号的识别
            # 待处理：如果模型输出\box, \text等latex符号，仍然无法正确提取
            expr_str = expr_str.strip()
            # 1) 去掉行间公式 $$…$$ 的定界符
            expr_str = re.sub(r'^\s*\$\$(.*?)\$\$\s*$', r'\1', expr_str)
            # 2) 去掉行内公式 $…$ 的定界符
            expr_str = re.sub(r'^\s*\$(.*?)\$\s*$', r'\1', expr_str)

            # 替换 \boxed{...} 为 ...
            expr_str = re.sub(r'\\boxed{([^}]+)}', r'\1', expr_str)
            # 替换 `\frac{a}{b}` 为 `a/b`
            expr_str = re.sub(r'\\frac{([^}]+)}{([^}]+)}', r'(\1)/(\2)', expr_str)
            # 替换 `\cdot` 为 `*`
            expr_str = expr_str.replace('\\cdot', '*')
            # 替换掉 LaTeX 中的转义字符 `\\` 为 `\`
            expr_str = expr_str.replace('\\', '')
            return expr_str

        def try_parse_expr(expr_str):
            """尝试将字符串转为 sympy 表达式"""
            try:
                # 预处理 LaTeX 表达式
                expr_str = preprocess_latex(expr_str)
                # 尝试使用 sympy 的 sympify 解析
                parsed = sympify(expr_str)
                # 返回简化后的表达式
                return simplify(parsed)
            except Exception as e:
                print(f"Parse failed for '{expr_str}': {e}")
                return None  # 返回 None 表示无法解析

        for expr_str in expr_list:
            parsed = try_parse_expr(expr_str)
            if parsed is None:
                result.append(expr_str)  # 无法解析，保留原始字符串
                continue

            found = False
            for key in simplified_map:
                try:
                    if simplify(parsed - key) == 0:
                        result.append(simplified_map[key])
                        found = True
                        break
                except Exception as e:
                    print(f"Comparison failed: parsed={parsed}, key={key}, error={e}")
            # 这一行在4.21进行了修改，防止有理数过长导致str操作爆掉
            if not found:
                try:
                    # 优先用精确表达式
                    canonical = str(parsed)
                except ValueError:
                    # 如果太大转不动，就退成浮点字符串
                    canonical = str(parsed.evalf(10))
                simplified_map[parsed] = canonical
                result.append(canonical)


        return result
    
    global_reward = 0.0
    correct_logp = 0.0
    num_generation = len(local_reward)

    for i in range(num_generation):
        correct_logp += local_reward[i]
    correct_logp = correct_logp / num_generation
    
    # canonical_exprs = canonicalize_expressions(extract_answer)
    entropy = entropy_from_list(extract_answer)

    # print("统一形式：", canonical_exprs)
    # print("熵值：", entropy)

    if reward_coef_flg:
        reward_coef =  ( 1 - lamda * entropy )
    else:
        reward_coef = 1
    global_reward = correct_logp * reward_coef

    return global_reward, entropy

#######
def compute_grpo_outcome_advantage_with_global(
                                   reward_coef_flg: bool,
                                   token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   index: torch.Tensor,extract_answer,output_file,
                                   epsilon: float = 1e-6,
                                   norm_adv_by_std_in_grpo: str = True,
                                   ):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # print('token_level_rewards.shape', token_level_rewards.shape)
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    global_rewards = torch.zeros_like(scores) # 144 * 1
    global_scores = torch.zeros_like(scores) # 144 * 1
    local_reward = torch.zeros_like(scores) # 144 * 1
    entropy = torch.zeros_like(scores)

    id2score = defaultdict(list)
    id2extract_answer = defaultdict(list)
    id2mean = {}
    id2std = {}
    id2global_score = {}
    id2entropy = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        # print("bsz", bsz)
        # print("index", index)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2extract_answer[index[i]].append(extract_answer[i])
            
        for idx in id2score:  # 对batch size中的每个index进行处理 24

            id2global_score[idx], id2entropy[idx] = compute_score_global(id2score[idx], id2extract_answer[idx], reward_coef_flg, output_file) ## 计算global reward

            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        global_scores_mean = torch.mean(torch.tensor(list(id2global_score.values())))
        global_scores_std = torch.std(torch.tensor(list(id2global_score.values())))
        for i in range(bsz):
            global_rewards[i] = id2global_score[index[i]]
            entropy[i] = id2entropy[index[i]]
            if norm_adv_by_std_in_grpo:
                local_reward[i] = scores[i]
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
                global_scores[i] = (id2global_score[index[i]] - global_scores_mean) / (global_scores_std + epsilon) 
            else:
                scores[i] = scores[i] - id2mean[index[i]]
                global_scores[i] = (id2global_score[index[i]] - global_scores_mean)
        
        scores = scores.unsqueeze(-1) * response_mask
        global_scores = global_scores.unsqueeze(-1) * response_mask
        # print("local_adv", scores)
    return scores, scores, global_scores, entropy, global_rewards, local_reward 

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6
):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask)

    return scores, scores


def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6
):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor
):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss

def compute_policy_loss_with_global(
                        local_reward,
                        global_flg,
                        global_loss_coef,
                        old_log_prob,
                        log_prob,
                        advantages,
                        response_mask,
                        global_adv,
                        reward_entropy,
                        cliprange=None,
                        cliprange_low=None,
                        cliprange_high=None,
                        clip_ratio_c=3.0,
                        loss_agg_mode="token-mean",
                        b=0.5,
                        k=1):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        use_token_level_loss: (bool)
            Whether to use token level loss
    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    ### global loss
    global_pg_losses1 = -global_adv * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    global_pg_losses2 = -global_adv * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    global_clip_pg_losses1 = torch.maximum(
        global_pg_losses1, global_pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    global_pg_clipfrac = verl_F.masked_mean(torch.gt(global_pg_losses2, global_pg_losses1).float(), response_mask)

    global_pg_losses3 = -global_adv * clip_ratio_c
    global_clip_pg_losses2 = torch.min(global_pg_losses3, global_clip_pg_losses1)
    global_pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(global_clip_pg_losses1, global_pg_losses3) * (global_adv < 0).float(), response_mask
    )

    global_pg_losses = torch.where(global_adv < 0, global_clip_pg_losses2, global_clip_pg_losses1)

    ## gather 2 loss
    global_loss_weight = torch.sigmoid(k * (reward_entropy - b))
    not_zero = (local_reward != 0).to(local_reward.dtype).unsqueeze(1).expand(-1, advantages.shape[1])

    # print("local_reward shape", local_reward.shape)
    # print("not_zero shape", not_zero.shape)
    # print("global_loss_weight shape", global_loss_weight.shape)
    
    # print("not_zero", not_zero)
    all_same = (advantages != 0).to(advantages.dtype)
    all_same = all_same[:, 0]
    
    if global_flg == 'no-global':
        policy_loss = pg_losses
    else:
        if global_flg == 'soft-with-forced':
            global_loss_weight = global_loss_weight * all_same
            policy_loss = global_pg_losses * (1 - global_loss_weight)[:, None] + pg_losses * global_loss_weight[:, None] ###
        elif global_flg == 'only-zero':
            global_loss_weight = not_zero
            policy_loss = global_pg_losses * (1 - global_loss_weight) + pg_losses * global_loss_weight
        elif global_flg == 'only-zero-wrong':
            global_loss_weight = not_zero
            policy_loss = global_pg_losses * (1 - global_loss_weight)[:, None] + pg_losses * global_loss_weight[:, None]
            print("policy_loss shape", policy_loss.shape)
        elif global_flg == 'soft-with-zero':
            global_loss_weight = global_loss_weight.unsqueeze(1) * not_zero
            policy_loss = global_pg_losses * (1 - global_loss_weight) + pg_losses * global_loss_weight ###
            print("policy_loss shape", policy_loss.shape)
        elif global_flg == 'only-forced':
            global_loss_weight = all_same
            policy_loss = global_pg_losses * (1 - global_loss_weight)[:, None] + pg_losses * global_loss_weight[:, None] ###
        elif global_flg == 'only-soft':
            policy_loss = global_pg_losses * (1 - global_loss_weight)[:, None] + pg_losses * global_loss_weight[:, None] ###
        elif global_flg == 'only-soft-with-coef':
            policy_loss = global_pg_losses * (1 - global_loss_weight)[:, None] * global_loss_coef + pg_losses * global_loss_weight[:, None] ###
        elif global_flg == 'only-global':
            policy_loss = global_pg_losses
        else:
            raise TypeError(f"global_flg only support: 'soft with forced', 'only forced', 'only soft', 'no global', but get: {global_flg}")
        
    pg_loss = agg_loss(loss_mat=policy_loss, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # global_pg_losses1 = -global_adv * ratio
    # global_pg_losses2 = -global_adv * torch.clamp(ratio, 1 - cliprange_low,
    #                                        1 + cliprange_high)
    # global_pg_losses = torch.maximum(global_pg_losses1, global_pg_losses2)
    # global_loss_weight = torch.sigmoid(k * (reward_entropy - b))
    # policy_loss = global_pg_losses * (1 - global_loss_weight)[:, None] + pg_losses * global_loss_weight[:, None] ###
    # policy_loss = pg_losses

    return pg_loss, pg_clipfrac, global_pg_clipfrac, ppo_kl, global_loss_weight, global_pg_losses, pg_losses, pg_clipfrac_lower, global_pg_clipfrac_lower

def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode="token-mean",
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" /
                                      "seq-mean-token-sum" /
                                      "seq-mean-token-mean" /
                                      "seq-mean-token-sum-norm" /
            "token-mean" is the default behavior

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
