"""
*
*
* File: utils.py
* Author: Fan Kai
* Soochow University
* Created: 2024-07-03 23:30:58
* ----------------------------
* Modified: 2024-07-06 09:22:09
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import os
from datetime import datetime
from enum import Enum


class PredictModelType(Enum):
    NN = "NN"
    BIM = "BIM"


def gen_save_dir(
    problem_name,
    algorithm_name,
    prefix="",
    model_type=PredictModelType.NN,
    NIND="",
    MAXGEN="",
    MAXTIME="",
):
    """生成 GA 运行存档目录

    Args:
        problem_name (str): 问题名称
        algorithm_name (str): 算法名称
        model_type (PredictModelType): 模型类型，默认为 PredictModelType.NN
        NIND (int): 种群规模
        MAXGEN (int): 最大进化代数
        MAXTIME (int): 最大运行时间 (s)

    Returns:
        str: 存档目录
    """
    _base_directory = os.path.join(os.getcwd(), "checkpoints/GA")

    # ------------------------- 存档文件名 ------------------------ #
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    dir_name = f"{prefix}_{problem_name}_{algorithm_name}_{model_type}_N{NIND}_G{MAXGEN}_T{MAXTIME}_{timestamp}".strip(
        "_"
    )

    # ------------------------ 构建存档目录 ------------------------ #
    archive_directory = os.path.join(_base_directory, dir_name).replace("\\", "/")
    os.makedirs(archive_directory, exist_ok=False)

    return archive_directory
