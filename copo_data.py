from huggingface_hub import HfApi, HfFolder, Repository
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="/Users/hjh/Downloads/data.zip",   # 本地 .zip 文件的完整路径
    path_in_repo="data.zip",               # 仓库中显示的路径/文件名，可自定义
    repo_id="Typiiing/copo_dataOnly",                      # 替换为你自己的用户名和仓库名
    repo_type="dataset",
    token="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"            # 你的 token
)
