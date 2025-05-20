save_path="/workspace/datasets/CO2/r1-vl/3b_instruct_grpo_only0KL" # 存储权重文件的路径
model_path="/workspace/datasets/LVLMs/Qwen/Qwen2.5-3B-Instruct" # 初始模型路径
step_start=1  # 默认为1，一般不用改
step_end=70  # 训练停止时的最终step

step_interval=1

for step in $(seq $step_start $step_interval $step_end);  do
  echo "Processing step $step..."
  python scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path ${model_path} \
    --local_dir ${save_path}/global_step_${step}/actor \
    --target_dir ${save_path}/global_step_${step}/huggingface
done