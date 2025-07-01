save_path=/cfs_turbo/alanhshao/copo/copo-code/outputs/0702/verl_7b_math_base_grpo  # 存储权重文件的路径
model_path=/home/tione/notebook/alanhshao/pretrained_models/Qwen2.5-7B-Instruct
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