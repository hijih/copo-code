set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY='01c8c25114910ecd17d17fa56bc8c837ac6aa85b'

#### change the following paths ####
code_path=/home/tione/notebook/alanhshao/copo/copo-code  # 放置代码的路径
data_path=/home/tione/notebook/alanhshao/copo/copo-data/data # 放置训练测试数据的路径
save_path=/cfs_turbo/alanhshao/copo/copo-code/outputs/0523/7b_instruct_grpo_dapo #"/workspace/datasets/CO2/r1-vl/3b_instruct_grpo_onlyGlobal_k5b1" # 存储权重文件的路径
model_path=/home/tione/notebook/alanhshao/pretrained_models/Qwen2.5-7B-Instruct # 7B模型需下载
wandb_name="verl_7b_instruct_grpo_dapo" # wandb实验名称

##### train ####
train_path=$data_path/DAPO-Math-17k/data/dapo-math-17k_0.1.parquet
train_files="['$train_path']"
aime24_test_path=$data_path/aime24/test_with_answer.parquet

test_files="['$aime24_test_path']"

loss_agg_mode="token-mean"
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 3))
overlong_penalty_factor=1.0

use_dynamic_bsz=True
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
gen_prompt_bsz=1536

python3 -m recipe.dapo.src.main_dapo \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=12 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.temperature=1.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_0.5b-instruct_dapo' \
    trainer.experiment_name=${wandb_name} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=1 \
    trainer.test_freq=0 \
    trainer.default_local_dir=${save_path} \
    trainer.total_epochs=1 $@
