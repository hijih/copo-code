#!/bin/bash
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY='01c8c25114910ecd17d17fa56bc8c837ac6aa85b'

#### change the following paths ####
num_gpus=4  ## 测试使用的gpu数量

code_path=/home/tione/notebook/alanhshao/copo/copo-code  # 放置代码的路径
data_path=/home/tione/notebook/alanhshao/copo/copo-data/data  # 放置训练测试数据的路径
save_path=/cfs_turbo/alanhshao/copo/copo-code/outputs/0524/3b_instruct_grpo_soft0K1b1 #"/cfs_turbo/alanhshao/copo/copo-code/outputs/0524/3b_instruct_grpo_soft0K5b1.5" # 存储推理数据及结果的路径

aime24_test_path=$data_path/aime24/test_with_answer.parquet
aime25_test_path=$data_path/aime25/test_formatOnly.parquet
math500_test_path=$data_path/MATH-500/test_formatOnly.parquet
gsm8k_test_path=$data_path/gsm8k/test_formatOnly.parquet

train_path="$data_path/MATH-500/test_formatOnly.parquet"
train_files="['$train_path']"

## test for gsm8k and math-500
test_files="['$math500_test_path', '$gsm8k_test_path']"

# 可以自由调整这几个变量：
step_start=1
step_end=70
step_interval=1

num_gpus=4
gpus_per_group=2
num_groups=$((num_gpus / gpus_per_group))

step_list=()
for step in $(seq $step_start $step_interval $step_end); do
    step_list+=($step)
done

total_steps=${#step_list[@]}
echo "Total effective steps: $total_steps"

steps_per_group=$(( (total_steps + num_groups - 1) / num_groups ))  

for group_id in $(seq 0 $((num_groups - 1))); do  ### 可以修改此处的起始值

    start_idx=$((group_id * steps_per_group))
    end_idx=$((start_idx + steps_per_group - 1))
    if [ $end_idx -ge $total_steps ]; then
        end_idx=$((total_steps - 1))
    fi

    gpu_steps=("${step_list[@]:$start_idx:$((end_idx - start_idx + 1))}")

    gpu_start=$((group_id * gpus_per_group))
    gpu_end=$((gpu_start + gpus_per_group - 1))
    gpu_list=$(seq -s, $gpu_start $gpu_end)

    echo "Group $group_id using GPUs $gpu_list for steps: ${gpu_steps[@]}"

    (
    for step in "${gpu_steps[@]}"; do
        echo "Launching step $step on GPU group $gpu_list"

        model_path="${save_path}/global_step_${step}/huggingface"
        experiment_name="${save_path}-${step}"
        local_dir="${save_path}/mg/${step}"

        CUDA_VISIBLE_DEVICES=$gpu_list \
        python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=grpo \
        custom_reward_function.path=$code_path/verl/utils/reward_score/multi-reward.py \
            data.train_files="$train_files" \
            data.val_files="$test_files" \
            data.train_batch_size=48 \
            data.max_prompt_length=2048 \
            data.max_response_length=2048 \
            data.filter_overlong_prompts=True \
            data.truncation='error' \
            actor_rollout_ref.model.path="$model_path" \
            actor_rollout_ref.actor.k=5 \
            actor_rollout_ref.actor.b=1.5 \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.clip_ratio=0.2 \
            actor_rollout_ref.actor.entropy_coeff=0 \
            actor_rollout_ref.actor.ppo_mini_batch_size=12 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
            actor_rollout_ref.rollout.n=6 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=6 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            actor_rollout_ref.rollout.free_cache_engine=true \
            actor_rollout_ref.rollout.temperature=1 \
            actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
            actor_rollout_ref.rollout.val_kwargs.top_p=1 \
            actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
            actor_rollout_ref.rollout.val_kwargs.n=8 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=true \
            algorithm.kl_ctrl.kl_coef=0 \
            trainer.critic_warmup=0 \
            trainer.logger=['console','wandb'] \
            trainer.project_name="test" \
            trainer.experiment_name="${experiment_name}" \
            trainer.n_gpus_per_node=2 \
            trainer.nnodes=1 \
            trainer.val_before_train=True \
            +trainer.val_only=True \
            trainer.save_freq=2 \
            trainer.test_freq=2 \
            trainer.default_local_dir="$local_dir" \
            trainer.total_epochs=1 $@
    done
    ) &
done

wait

## test for aime24 and aime25
test_files="['$aime24_test_path', '$aime25_test_path']"

# 可以自由调整这几个变量：
step_start=1
step_end=70
step_interval=1

num_gpus=4
gpus_per_group=2
num_groups=$((num_gpus / gpus_per_group))

step_list=()
for step in $(seq $step_start $step_interval $step_end); do
    step_list+=($step)
done

total_steps=${#step_list[@]}
echo "Total effective steps: $total_steps"

steps_per_group=$(( (total_steps + num_groups - 1) / num_groups ))  

for group_id in $(seq 0 $((num_groups - 1))); do  ### 可以修改此处的起始值

    start_idx=$((group_id * steps_per_group))
    end_idx=$((start_idx + steps_per_group - 1))
    if [ $end_idx -ge $total_steps ]; then
        end_idx=$((total_steps - 1))
    fi

    gpu_steps=("${step_list[@]:$start_idx:$((end_idx - start_idx + 1))}")

    gpu_start=$((group_id * gpus_per_group))
    gpu_end=$((gpu_start + gpus_per_group - 1))
    gpu_list=$(seq -s, $gpu_start $gpu_end)

    echo "Group $group_id using GPUs $gpu_list for steps: ${gpu_steps[@]}"

    (
    for step in "${gpu_steps[@]}"; do
        echo "Launching step $step on GPU group $gpu_list"

        model_path="${save_path}/global_step_${step}/huggingface"
        experiment_name="${save_path}-${step}"
        local_dir="${save_path}/aime/${step}"

        CUDA_VISIBLE_DEVICES=$gpu_list \
        python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=grpo \
        custom_reward_function.path=$code_path/verl/utils/reward_score/multi-reward.py \
            data.train_files="$train_files" \
            data.val_files="$test_files" \
            data.train_batch_size=48 \
            data.max_prompt_length=2048 \
            data.max_response_length=2048 \
            data.filter_overlong_prompts=True \
            data.truncation='error' \
            actor_rollout_ref.model.path="$model_path" \
            actor_rollout_ref.actor.k=5 \
            actor_rollout_ref.actor.b=1.5 \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.clip_ratio=0.2 \
            actor_rollout_ref.actor.entropy_coeff=0 \
            actor_rollout_ref.actor.ppo_mini_batch_size=12 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
            actor_rollout_ref.rollout.n=6 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=6 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            actor_rollout_ref.rollout.free_cache_engine=true \
            actor_rollout_ref.rollout.temperature=1 \
            actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
            actor_rollout_ref.rollout.val_kwargs.top_p=1 \
            actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
            actor_rollout_ref.rollout.val_kwargs.n=64 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=true \
            algorithm.kl_ctrl.kl_coef=0 \
            trainer.critic_warmup=0 \
            trainer.logger=['console','wandb'] \
            trainer.project_name="test" \
            trainer.experiment_name="${experiment_name}" \
            trainer.n_gpus_per_node=2 \
            trainer.nnodes=1 \
            trainer.val_before_train=True \
            +trainer.val_only=True \
            trainer.save_freq=2 \
            trainer.test_freq=2 \
            trainer.default_local_dir="$local_dir" \
            trainer.total_epochs=1 $@
    done
    ) &
done

wait