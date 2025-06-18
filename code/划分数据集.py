import pandas as pd
import numpy as np

# 读取csv数据集
df = pd.read_csv('/home/workspace/lgq/data/全新训练数据20250613.csv')

# 随机打乱数据
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 按1:9比例划分
n_total = len(df_shuffled)
n_part1 = int(n_total * 0.1)
n_part2 = n_total - n_part1

df_part1 = df_shuffled.iloc[:n_part1]
df_part2 = df_shuffled.iloc[n_part1:]

# 保存为新的csv文件
df_part1.to_csv('/home/workspace/lgq/data/全新训练数据20250613_10percent.csv', index=False)
df_part2.to_csv('/home/workspace/lgq/data/全新训练数据20250613_90percent.csv', index=False)
