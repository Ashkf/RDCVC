"""
* dataset for R&D Cleanroom Ventilation Control
*
* File: rdcvc_dataset.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-19 03:04:13
* ----------------------------
* Modified: 2024-06-01 15:33:44
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import dataset


class RDCVCDataset(dataset.Dataset):
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
        _data_mark = [
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
        _target_mark = [
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

        # ----------------------- read data ---------------------- #
        df = pd.read_csv(args.train_path) if is_train else pd.read_csv(args.eval_path)
        _data = torch.tensor(df[_data_mark].values, dtype=torch.float32)
        _target = torch.tensor(df[_target_mark].values, dtype=torch.float32)

        # ------------------------ zscore ------------------------ #
        self.data = self.scaler.scale(_data, "x", self.is_train).to(self.device)
        self.target = self.scaler.scale(_target, "y", self.is_train).to(self.device)

        self.num_samples = len(_data)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return self.data[index], self.target[index]

    def __len__(self) -> int:
        return self.num_samples
