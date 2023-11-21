import numpy as np
from pyDOE2 import lhs

# 定义输入维度
n_dim = 3
# 定义采样点数
n_samples = 100

# 定义每个维度的范围
opening_range = [0, 50]
bounds = [opening_range] * n_dim  # 输入范围

# 生成 LHS 样本
samples_normalized = lhs(n_dim, samples=n_samples, criterion='corr')

samples = np.zeros_like(samples_normalized)  # 初始化样本
# 将样本范围映射到输入空间
for i in range(n_dim):
    samples[:, i] = np.round(
        samples_normalized[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0])

# samples = np.round(samples / 5) * 5  # 近似至 5 的倍数

np.savetxt(r'C:\Users\KAI\Desktop\lhs7.csv', samples, delimiter=',')  # 保存至桌面 csv 文件
print(samples)
