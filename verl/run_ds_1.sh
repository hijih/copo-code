set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,3,5
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=offline
# export WANDB_BASE_URL=https://api.bandw.top
# export WANDB_API_KEY=760a6965ccab99b933ed3402f7c28958502073dc
#### change the following paths ####
code_path=/workspace/code/copo-code  # 放置代码的路径
data_path=/workspace/datasets/hjh  # 放置训练测试数据的路径
save_path="/workspace/datasets/CO2/r1-vl/ds_7b_grpo" # 存储权重文件的路径
model_path="/workspace/datasets//LVLMs/DeepSeek-R1-Distill-Qwen-7B" # 初始模型路径（需下载）
wandb_name="verl_ds_7b_grpo" # wandb实验名称

##### train ####
train_path=$data_path/DAPO-Math-17k/data/dapo-math-17k_0.02.parquet
train_files="['$train_path']"
gsm8k_test_path=$data_path/gsm8k/test_formatOnly.parquet
aime24_test_path=$data_path/aime24/test_with_answer.parquet
test_files="['$gsm8k_test_path']"
loss_agg_mode="seq-mean-token-mean"
global_flg="no-global"
# VLLM_SWAP_GB=32
## global_flg: 'soft-with-forced', 'only-forced', 'only-soft', 'no-global', 'soft-with-zero'

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=$code_path/verl/utils/reward_score/multi-reward.py \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    +data.trust_remote_code=true \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.reward_coef_flg=false \
    actor_rollout_ref.actor.k=20 \
    actor_rollout_ref.actor.b=0.5 \
    actor_rollout_ref.actor.global_flg=${global_flg} \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=12 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.temperature=1.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_1.5b-instruct_dapo' \
    trainer.experiment_name=${wandb_name} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=1 \
    trainer.test_freq=0 \
    trainer.default_local_dir=${save_path} \
    trainer.total_epochs=1 $@
