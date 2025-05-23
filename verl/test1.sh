set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

#### change the following paths ####
code_path=/workspace/code/copo-code  # 放置代码的路径
data_path=/workspace/datasets/hjh  # 放置训练测试数据的路径
save_path="/workspace/datasets/CO2/r1-vl/3b_instruct_grpo_soft0K5b1" # 存储推理数据及结果的路径

step_start=1
step_end=70
step_interval=1

aime24_test_path=$data_path/aime24/test_with_answer.parquet
aime25_test_path=$data_path/aime25/test_formatOnly.parquet
math500_test_path=$data_path/MATH-500/test_formatOnly.parquet
gsm8k_test_path=$data_path/gsm8k/test_formatOnly.parquet

train_path="$data_path/MATH-500/test_formatOnly.parquet"
train_files="['$train_path']"

test_files="['$math500_test_path', '$gsm8k_test_path']"


for step in $(seq $step_start $step_interval $step_end); do
    echo "Processing step $step..."

    model_path="${save_path}/global_step_${step}/huggingface"
    experiment_name="${save_path}-${step}"
    local_dir="${save_path}/mg/${step}"

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
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.val_before_train=True \
        +trainer.val_only=True \
        trainer.save_freq=2 \
        trainer.test_freq=2 \
        trainer.default_local_dir="$local_dir" \
        trainer.total_epochs=1 $@
done


test_files="['$aime24_test_path', '$aime25_test_path']"

for step in $(seq $step_start $step_interval $step_end); do
    echo "Processing step $step..."

    model_path="${save_path}/global_step_${step}/huggingface"
    experiment_name="${save_path}-${step}"
    local_dir="${save_path}/aime/${step}"

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
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.val_before_train=True \
        +trainer.val_only=True \
        trainer.save_freq=2 \
        trainer.test_freq=2 \
        trainer.default_local_dir="$local_dir" \
        trainer.total_epochs=1 $@
done