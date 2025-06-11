# COPO-code



## 环境配置

* 代码地址：
* 镜像及数据存储地址： https://huggingface.co/datasets/Typiiing/copo-data
* Docker 镜像：verl_bg_img.tar
* 训练测试数据：data.zip

## 代码

* 代码文件架构

```
copo-code/
├── verl
│   ├── run1.sh
│   ├── merge_all.sh
│   ├── test_mg.sh
│   ├── test_aime.sh
│   ...
```

## 运行

### **1. 训练**

- 进入 docker 镜像，在镜像中进入对应的`copo-code/verl`路径
- 修改`copo-code/verl/run1.sh`中的下列路径：

```sh
#### change the following paths ####

code_path=/workspace/code/copo-code  # 放置代码的路径
data_path=/workspace/datasets/hjh  # 放置训练测试数据的路径
save_path="/workspace/datasets/CO2/r1-vl/3b_instruct_grpo_only0KL" # 存储权重文件的路径
model_path="/workspace/datasets/LVLMs/Qwen/Qwen2.5-3B-Instruct" # 初始模型路径（需下载）
wandb_name="verl_3b_instruct_grpo_only0KL" # wandb实验名称
```

- 运行 run1.sh 进行模型训练，训练至65~70 step，即可手动停止

### **2. 测试**

- 进入 docker 镜像，在镜像中进入对应的`copo-code/verl`路径
- 修改`copo-code/verl/merge_all.sh`中的下列内容：

```sh
save_path="/workspace/datasets/CO2/r1-vl/3b_instruct_grpo_only0KL" # 存储权重文件的路径
model_path="/workspace/datasets/LVLMs/Qwen/Qwen2.5-3B-Instruct" # 初始模型路径
step_start=1  # 默认为1，一般不用改
step_end=70  # 训练停止时的最终step
```

- 在镜像中运行 merge_all.sh，将分布式存储的模型转化为 huggingface 格式
- 修改 test_mg.sh 和 test_aime.sh 中的下列内容：

```sh
code_path=/workspace/code/copo-code  # 放置代码的路径
data_path=/workspace/datasets/hjh  # 放置训练测试数据的路径
save_path="/workspace/datasets/CO2/r1-vl/3b_instruct_grpo_only0KL" # 存储推理数据及结果的路径

step_start=1 # 测试的起始路径
step_end=20 # 测试的终止路径
```

- 在镜像中运行 test_mg.sh 和 test_aime.sh，进行测试
- test_mg.sh 逐 step 串行运行较慢，可再复制几个脚本，修改 step_start, step_end 分段并行。之前一般分为3个脚本运行，即step1~22, step23~44, step45~65。如果有更多卡，test_aime.sh 也可以这样做 : )
- 运行结束后，推理数据及结果会放在上面设置的 save_path 里，麻烦将 save_path 的所有内容进行上传，谢谢~

### 结果文件
- [05.22] [https://huggingface.co/datasets/shaohang/COPO/outputs_0522.tar](https://huggingface.co/datasets/shaohang/COPO/blob/main/outputs_0522.tar)
- [05.23] [https://huggingface.co/datasets/shaohang/COPO/outputs_0523.tar](https://huggingface.co/datasets/shaohang/COPO/blob/main/outputs_0523.tar)
- [05.24-3b] [https://huggingface.co/datasets/shaohang/COPO/outputs_0524_3b.tar](https://huggingface.co/datasets/shaohang/COPO/blob/main/outputs_0524_3b.tar)
- [05.24-7b] [https://huggingface.co/datasets/shaohang/COPO/outputs_0524_7b.tar](https://huggingface.co/datasets/shaohang/COPO/blob/main/outputs_0524_7b.tar)
- [05.27] [https://huggingface.co/datasets/shaohang/COPO/outputs_0527.tar](https://huggingface.co/datasets/shaohang/COPO/blob/main/outputs_0527.tar)
- [05.29] [https://huggingface.co/datasets/shaohang/COPO/outputs_0529.tar](https://huggingface.co/datasets/shaohang/COPO/blob/main/outputs_0529.tar)
- [06.03 and 06.04] [https://huggingface.co/datasets/shaohang/COPO/outputs_0603_and_0604.tar](https://huggingface.co/datasets/shaohang/COPO/blob/main/outputs_0603_and_0604.tar)
