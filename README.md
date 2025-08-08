# COPO-code



## Experiment

* Our code is trained based on the VERL framework, and the training environment setup follows the guidelines provided in the official VERL documentation.

## Code

```
copo-code/
├── verl
│   ├── run_copo.sh
│   ...
```

## Running

- Change the following paths in `copo-code/verl/run_copo.sh`:

```sh
#### change the following paths ####
code_path=/workspace/code/copo-code  # path to the code
data_path=/workspace/datasets/your_dataset  # path to the test data
save_path="/workspace/datasets/your_save_path" # path to save the model
model_path="/workspace/datasets/your_model_path" # path to the model
wandb_name="your_experiment_name" # name of wandb experiment
project_name="your_project_name" # name of wandb project
```

- These parameters can be adjusted to accommodate training on different datasets and to explore various soft blending hyperparameter settings:

```sh
train_path=$data_path/DAPO-Math-17k/data/dapo-math-17k.parquet # train dataset path 
train_files="['$train_path']"
math500_test_path=$data_path/math500.parquet # test dataset path with prompt
aime24_test_path=$data_path/aime24.parquet
test_files="['your_test_path']"
global_flg="soft-with-zero" # soft-with-zero represents our COPO method 
soft_blending_k=5 # soft blending hyperparameter gamma
soft_blending_b=1 # soft blending hyperparameter rho
```

- Run bash `verl/run_copo.sh` to train the model with COPO
