import json
import pandas as pd

# 设置统一的名称和文件路径
name = "3b_k10b1.5"
jsonl_paths = [
    '/Users/hjh/Desktop/python/result.jsonl',
    '/Users/hjh/Desktop/python/result 2.jsonl'
]
output_xlsx = "/Users/hjh/Desktop/" + name + "_output.xlsx"
target_datasets = ["aime24", "aime25", "MATH-500", "gsm8k"]

# metric 映射
metric_pairs = [
    [("mean@64", "mean@64"), ("maj@64/mean", "maj@64")],
    [("mean@8", "mean@8"), ("maj@8/mean", "maj@8")]
]

all_rows = []

for jsonl_path in jsonl_paths:
    rows = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            step = record.get("step")
            found_datasets = []

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

    df = pd.DataFrame(rows)
    df["metric"] = df["metric"].replace({
        "mean@64": "b_mean",
        "mean@8": "b_mean",
        "maj@64": "b_maj64_mean",
        "maj@8": "b_maj8_mean"
    })

    all_rows.append(df)

# 合并所有数据并写入 Excel（一个 sheet，无表头重复）
final_df = pd.concat(all_rows, ignore_index=True)
final_df.to_excel(output_xlsx, index=False)

print("✅ 两个文件已成功合并输出为 Excel：", output_xlsx)