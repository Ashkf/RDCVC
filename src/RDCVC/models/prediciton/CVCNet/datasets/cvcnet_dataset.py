"""
* dataset for CVCNet
*
* File: bim_cpn_dataset.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-19 03:04:13
* ----------------------------
* Modified: 2024-06-01 13:03:00
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

from typing import Dict

import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import dataset


class CVCNetDataset(dataset.Dataset):  # 注意父类的名称，不能写 dataset
    def __init__(self, args, is_train=True, scaler=None, device=None):
        """
            一般用来 train 的预处理和 test 的预处理是不同的，需要区分二者的参数。

        Args:
            scaler: 归一化器
            args: 参数列表
            is_train: 是否是训练集
        """
        super().__init__()
        self.scaler = scaler
        self.is_train = is_train  # 是否是训练集
        self.device = device
        self.normalize_target = args.normalize_target  # 归一化的目标
        self.normalize_method = args.normalize_method  # 归一化的方法
        self.device = args.device[0]
        self.target_key = [
            "TOT_FRSH_VOL",
            "TOT_SUPP_VOL",
            "TOT_EXH_VOL",
            "TOT_RET_VOL",
            "RM1_PRES",  # a
            "RM2_PRES",  # b
            "RM3_PRES",  # d
            "RM4_PRES",  # e
            "RM5_PRES",  # f
            "RM6_PRES",  # c
        ]
        self.data_key = [
            "MAU_FREQ",
            "AHU_FREQ",
            "EF_FREQ",
            "RM1_SUPP_DMPR_0",
            "RM2_SUPP_DMPR_0",
            "RM3_SUPP_DMPR_0",
            "RM4_SUPP_DMPR_0",
            "RM5_SUPP_DMPR_0",
            "RM6_SUPP_DMPR_0",
            "RM6_SUPP_DMPR_1",
            "RM2_RET_DMPR_0",
            "RM3_RET_DMPR_0",
            "RM4_RET_DMPR_0",
            "RM6_RET_DMPR_0",
            "RM3_EXH_DMPR_0",
            "RM4_EXH_DMPR_0",
            "RM5_EXH_DMPR_0",
            "RM5_EXH_DMPR_1",
        ]

        # ----------------------- read data ---------------------- #
        if is_train:
            df = pd.read_csv(args.train_path)
        else:
            df = pd.read_csv(args.eval_path)
        _data = df[self.data_key + self.target_key].copy()
        self.data: Dict[str, Tensor] = self.preproces_data(_data)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        """Retrieves the data at the specified index.

        Args:
            index (int): The index of the data to retrieve.

        Returns:
            tuple[Tensor, Tensor]: A tuple of the input data and target variables.
        """
        return self.data["data"][index], self.data["target"][index]

    def __len__(self) -> int:
        """返回数据集的大小"""
        return len(self.data["data"])

    def preproces_data(self, data: DataFrame) -> Dict[str, Tensor]:
        """Preprocesses the input data and target variables.

        use sklearn's StandardScaler to normalize the data

        This process loads all data into the device's memory. (cuda)

        Args:
            data (DataFrame): The input data.

        Returns:
            Dict[str, Tensor]: A tuple of the input data and target variables.
        """
        # ------------------------ unpack ------------------------ #
        _data: DataFrame = data[self.data_key].copy()
        _target: DataFrame = data[self.target_key].copy()

        # ------------------------ to tensor --------------------- #
        _data = torch.tensor(_data.values, dtype=torch.float32).to(self.device)
        _target = torch.tensor(_target.values, dtype=torch.float32).to(self.device)

        # ----------------------- normalize ---------------------- #
        _data: Tensor = self.scaler.scale(_data, "x", self.is_train)
        _target: Tensor = self.scaler.scale(_target, "y", self.is_train)

        return {"data": _data.to(self.device), "target": _target.to(self.device)}
