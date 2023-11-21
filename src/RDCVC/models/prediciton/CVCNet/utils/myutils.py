"""
*
*
* File: utils.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-19 03:05:38
* ----------------------------
* Modified: 2023-11-19 03:05:48
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


def readCSV(file_path, cols=None, num_skip=0) -> np.ndarray:
    """从 csv 文件读取数据

    Args:
        file_path (str):
        cols (list):读取的列的列表（从 0 开始）
        num_skip (int):跳过的行数（跳过表头）

    Returns:
        ndarry: data frome csv file
    """
    if cols is None:
        cols = [0]
    data = np.loadtxt(
        open(file_path, "rb"), delimiter=",", skiprows=num_skip, usecols=cols
    )
    return data


def randfloat(left, right, num):
    """指定范围内生成指定个数的随机数

    Args:
        left (_type_): 范围左区间
        right (_type_): 范围右区间
        num (int): 随机数个数

    Returns:
        list: 随机数列表
    """
    if left > right:
        return None
    else:
        _len = right - left
        b = right - _len
        out = (np.random.rand(num) * _len + b).tolist()
        out = np.array(out)
        return out.tolist()


def PlotLoss(num_epoch, loss_list, label):
    plt.figure()
    plt.subplot(111)
    plt.plot([i + 1 for i in range(1, num_epoch)], loss_list, label=label)
    plt.title("loss")
    plt.legend()
    plt.show()


def save(model, model_filename):
    time_now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    file_name = model_filename + "_" + time_now
    dir_path = r"./checkpoints"  # 获取 checkpoints 路径
    file_path = os.path.join(dir_path, file_name)
    # 只保存网络中的参数 (速度快，占内存少)
    torch.save(model.state_dict(), f"{file_path}.pth")


# 计时器
def littletimer(func):
    def func_wrapper(*args, **kwargs):
        from time import time

        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print(f"[time cost]    <{func.__name__:^20}>    cost time {time_spend:.3f} s")
        return result

    return func_wrapper


def get_now_time():
    """获取当前时间，格式为：2020-01-01T00-00-00

    Returns:
        str: 当前时间
    """
    return time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())


def set_random_seed(seed):
    if seed == 1234:
        # 1234 是默认值，不需要设置
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True


def get_time():
    """获取当前时间

    Returns:
        str: 当前时间 (20201231_235959)
    """
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())
