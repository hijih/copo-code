import json
import pandas as pd
import re

name = 'onlyGlobal'

jsonl_path = '/Users/hjh/Downloads/outputs0522/mg/result.jsonl'
output_xlsx = "/Users/hjh/Desktop/results/onlyGlobal_output.xlsx"
target_datasets = ["aime24", "aime25", "MATH-500", "gsm8k"]

# 定义 metric 映射和优先级
metric_pairs = [
    [("mean@64", "mean@64"), ("maj@64/mean", "maj@64")],
    [("mean@8", "mean@8"), ("maj@8/mean", "maj@8")]
]

rows = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        step = record.get("step")

        found_datasets = []

        # 筛选出含有至少一个 mean 和一个 maj metric 的数据集
        for dataset in target_datasets:
            has_metrics = []
            for raw_m, std_m in sum(metric_pairs, []):
                for key in record:
                    if "val-aux" in key and dataset in key and raw_m in key:
                        has_metrics.append(std_m)
                        break
            if "mean@64" in has_metrics and "maj@64" in has_metrics:
                found_datasets.append((dataset, "64"))
            elif "mean@8" in has_metrics and "maj@8" in has_metrics:
                found_datasets.append((dataset, "8"))

            if len(found_datasets) == 2:
                break

        # 只处理选出的两个数据集
        for dataset, precision in found_datasets[:2]:
            for raw_m, std_m in metric_pairs[0 if precision == "64" else 1]:
                for key, value in record.items():
                    if "val-aux" in key and dataset in key and raw_m in key:
                        rows.append({
                            'Name': name,
                            "step": step,
                            "data": dataset,
                            "metric": std_m,
                            "value": value
                        })
                        break

# 创建 DataFrame 并重命名 metric 字段
df = pd.DataFrame(rows)
df["metric"] = df["metric"].replace({
    "mean@64": "b_mean",
    "mean@8": "b_mean",
    "maj@64": "b_maj64_mean",
    "maj@8": "b_maj8_mean"
})

# 写入 Excel
df.to_excel(output_xlsx, index=False)
print("✅ 每条数据成功转换，已保存为 Excel。")