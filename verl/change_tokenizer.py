from transformers import AutoConfig
from transformers import AutoTokenizer

for i in range(1, 70):
    path = '/home/tione/notebook/alanhshao/copo/copo-code/outputs/0524/7b_instruct_grpo/global_step_'+str(i)+'/huggingface'
    config = AutoConfig.from_pretrained("/home/tione/notebook/alanhshao/pretrained_models/Qwen2.5-7B-Instruct")
    config.save_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained('/home/tione/notebook/alanhshao/pretrained_models/Qwen2.5-7B-Instruct')
    tokenizer.save_pretrained(path)
    print(i)