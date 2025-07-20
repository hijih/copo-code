import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取 Excel 文件
file_path = "/Users/hjh/Desktop/results/new/1.5b_base/merged.xlsx"
df = pd.read_excel(file_path)

# 创建输出文件夹
output_dir = "1.5b-math-base-new"
os.makedirs(output_dir, exist_ok=True)

# 存储每个 metric 各 Name 的最大值
summary_records = []

# 遍历每个 data 的取值
for data_val in df['data'].unique():
    df_data = df[df['data'] == data_val]

    # 遍历该 data 下的每个 metric
    for metric_val in df_data['metric'].unique():
        df_metric = df_data[df_data['metric'] == metric_val]

        # 创建画布
        plt.figure(figsize=(10, 6))

        # 按 Name 分组并绘图
        for name, group in df_metric.groupby('Name'):
            group_sorted = group.sort_values(by='step')
            plt.plot(group_sorted['step'], group_sorted['value'], label=name)

            # 计算最大值和 step
            max_row = group_sorted.loc[group_sorted['value'].idxmax()]
            max_step = max_row['step']
            max_value = max_row['value']

            # 打印到终端
            print(f"[data: {data_val} | metric: {metric_val} | Name: {name}] Max Value = {max_value} at Step = {max_step}")

            # 添加到统计结果
            summary_records.append({
                'data': data_val,
                'metric': metric_val,
                'Name': name,
                'max_value': max_value,
                'max_step': max_step
            })

        plt.title(f"data = {data_val}, metric = {metric_val}")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend(title='Name')
        plt.grid(True)

        # 保存图像
        filename = f"{data_val}_{metric_val}.png".replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

# 保存统计汇总为 CSV
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(os.path.join(output_dir, "max_values_summary.csv"), index=False)
print("\n📄 每个metric下各Name的最大值已保存为 'plots/max_values_summary.csv'")

# 计算每个 (data, metric) 的最大值中的最大值
top_max_summary = summary_df.loc[summary_df.groupby(['data', 'metric'])['max_value'].idxmax()]

# 保存结果
top_max_summary.to_csv(os.path.join(output_dir, "top_max_per_data_metric.csv"), index=False)
print("🌟 每个(data, metric)下最大值中的最大值已保存为 'plots/top_max_per_data_metric.csv'")