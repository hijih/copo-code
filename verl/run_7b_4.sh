set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_BASE_URL=https://api.bandw.top
export WANDB_API_KEY='01c8c25114910ecd17d17fa56bc8c837ac6aa85b'
#### change the following paths ####
code_path=/home/tione/notebook/alanhshao/copo/copo-code  # 放置代码的路径
data_path=/home/tione/notebook/alanhshao/copo/copo-data/data # 放置训练测试数据的路径
save_path=/cfs_turbo/alanhshao/copo/copo-code/outputs/0702/verl_7b_math_base_onlyGlobal
model_path=/home/tione/notebook/alanhshao/pretrained_models/Qwen2.5-Math-7B # Qwen2.5-Math-7B
wandb_name="verl_7b_math_base_onlyGlobal" # wandb实验名称

##### train ####
train_path=$data_path/DAPO-Math-17k/data/dapo-math-17k_0.1.parquet
train_files="['$train_path']"
aime24_test_path=$data_path/aime24/test_with_answer.parquet
test_files="['$aime24_test_path']"
loss_agg_mode="seq-mean-token-mean"
global_flg="only-global"

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
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=12 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.temperature=1.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_7b_math_base_dapo' \
    trainer.experiment_name=${wandb_name} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=1 \
    trainer.test_freq=0 \
    trainer.default_local_dir=${save_path} \
    trainer.total_training_steps=70 \
    trainer.total_epochs=1 $@