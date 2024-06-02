"""
*
*
* File: iotdp_dataset.py
* Author: Fan Kai
* Soochow University
* Created: 2024-05-31 11:50:39
* ----------------------------
* Modified: 2024-06-01 13:03:43
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import dataset


class IoTDamperDataset(dataset.Dataset):
    def __init__(self, args, is_train=True, scaler=None, device=None):
        super().__init__()
        self.scaler = scaler
        self.is_train = is_train
        self.device = device
        self.normalize_target = args.normalize_target  # 归一化的目标
        self.normalize_method = args.normalize_method  # 归一化的方法

        # 获取数据文件 path
        if is_train:
            df = pd.read_excel(args.train_path, sheet_name="train")
        else:
            df = pd.read_excel(args.eval_path, sheet_name="val")

        _data_mark = ["前阀片组开度", "后阀片开度", "风阀读取压差（Pa）"]
        _target_mark = [
            "风量罩风量（m3/h）",
        ]

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
