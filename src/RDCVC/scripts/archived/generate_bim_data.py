"""
从 bim 中生成数据
"""
import os

print(os.getcwd())

import time

import numpy as np
from pyDOE2 import lhs
from tqdm import tqdm

from CoreGA.utils.DampersProcessor import DampersProcessor
from utils.SQLiteManager import SQLiteManager
from CoreGA.utils.DataProcessor import DataProcessor


def generate_samples_LHS(n_samples):
    """按照 LHS 方法生成采样点"""

    n_dim = 18  # 定义输入维度

    # 定义每个维度的范围
    freq_range = [1, 500]  # 风机频率（/0.1Hz）
    opening_range = [0, 90]  # 阀门开度（/1°）
    bounds = [freq_range] * 3 + [opening_range] * 15

    # 生成 LHS 采样样本
    samples = lhs(n_dim, samples=n_samples, criterion='corr')

    # 将样本范围映射到输入空间
    for i in range(n_dim):
        samples[:, i] = np.round(
            samples[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][
                0]).astype(int)

    return samples


def experiment():
    """实验并生成数据

    在 bim 中实验并生成数据，若数据已存在，则跳过该条样本。
    """
    samples_path = "data/BIM_input_eval.csv"
    database_path = "data/SMKNRDC_bim2NN_20230708.db"
    table_name = "bim2NN_eval_data"
    num_samples = 1000

    # 尝试读取输入样本 从 data/train_input.csv
    try:
        samples = np.loadtxt(samples_path, delimiter=',', dtype=int)
        if len(samples) != num_samples:
            raise ValueError
    except OSError:
        # 生成输入样本，保存至 data/train_inpute.csv
        samples = generate_samples_LHS(num_samples)
        np.savetxt(samples_path, samples, delimiter=',', fmt='%d')

    # 初始化数据库
    dbm = SQLiteManager(database_path)
    dbm.create_table(table_name)
    dp = DampersProcessor()

    # 找到数据库中已有的样本的索引
    num_tested = dbm.get_data_count(table_name)
    print(f"已有 {num_tested} 条数据")
    time.sleep(1)

    start = time.time()  # 计时
    for i in tqdm(range(len(samples))):
        if i < num_tested:
            # 若该样本已存在，则跳过
            continue

        X_test = {"MAU_FREQ": samples[i][0] / 10,
                  "AHU_FREQ": samples[i][1] / 10,
                  "EF_FREQ": samples[i][2] / 10,
                  "RM1_SUPP_DMPR_0": samples[i][3],
                  "RM2_SUPP_DMPR_0": samples[i][4],
                  "RM3_SUPP_DMPR_0": samples[i][5],
                  "RM4_SUPP_DMPR_0": samples[i][6],
                  "RM5_SUPP_DMPR_0": samples[i][7],
                  "RM6_SUPP_DMPR_0": samples[i][8],
                  "RM6_SUPP_DMPR_1": samples[i][9],
                  "RM2_RET_DMPR_0": samples[i][10],
                  "RM3_RET_DMPR_0": samples[i][11],
                  "RM4_RET_DMPR_0": samples[i][12],
                  "RM6_RET_DMPR_0": samples[i][13],
                  "RM3_EXH_DMPR_0": samples[i][14],
                  "RM4_EXH_DMPR_0": samples[i][15],
                  "RM5_EXH_DMPR_0": samples[i][16],
                  "RM5_EXH_DMPR_1": samples[i][17]
                  }
        result = DataProcessor.parse_bim_result(dp.compute(X_test))
        dbm.add_data(result, table_name=table_name)
    end = time.time()
    print(f'耗时：{end - start} s')


if __name__ == "__main__":
    experiment()
