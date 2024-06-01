"""
* read excel files from directory, split train set and val set, save to excel
* 
* File: integrate_data.py
* Author: Fan Kai
* Soochow University
* Created: 2023-10-09 09:42:54
* ----------------------------
* Modified: 2023-10-09 10:30:21
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import os

import pandas as pd

directory_path = r"./data/IoT-Damper_v5_320250"  # 输入目录路径
output_path = os.path.join(directory_path, "IoTDamper.xlsx")  # 输出路径
# 划出验证集
num_spacelines = 6
val_indexs = [9, 15, 21, 23, 30, 32, 37, 44, 49]




# --------------------- read excel files -------------------- #


def read_excel_files(directory, num_skiprows):
    """读取目录下所有 excel 文件

    Args:
        directory (str): 目录路径
        num_skiprows (int): 跳过的行数

    Returns:
        Dataframe: 所有 excel 文件的数据
    """
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(directory, filename)
            df = pd.read_excel(file_path, skiprows=num_skiprows)
            all_data.append(df)
    return all_data


dfs = read_excel_files(directory_path, num_skiprows=4)


val_indexs = [ids - num_spacelines for ids in val_indexs]


def split_set(dfs, val_indexs):
    """从 dfs 中分离出验证集

    Args:
        dfs (list[Dataframe]): 所有数据
        val_indexs (list): 验证集索引

    Returns:
        df_train(Dataframe): 训练集
        df_val(Dataframe): 验证集
    """
    df_val = pd.DataFrame()
    df_train = pd.DataFrame()

    for df in dfs:
        val_rows = df[df.index.isin(val_indexs)]  # 提取索引行
        train_rows = df[~df.index.isin(val_indexs)]  # 提取剩余行

        df_val = pd.concat([df_val, val_rows])  # 将索引行添加到 df_val
        df_train = pd.concat([df_train, train_rows])  # 将剩余行添加到 df_train

    return df_train, df_val


df_train, df_val = split_set(dfs, val_indexs)

# --------------------- save to excel -------------------- #
writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
# train sheet
df_train.to_excel(writer, sheet_name="train", index=False)
# val sheet
df_val.to_excel(writer, sheet_name="val", index=False)
writer.close()
