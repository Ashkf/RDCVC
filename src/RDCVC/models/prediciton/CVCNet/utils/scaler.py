"""
提供 Scaler 类，用于数据归一化
"""

import os
import pickle
from enum import Enum

import torch
from torch import Tensor

from .standard_scaler_torch import StandardScalerTorch


class ScalerMode(Enum):
    NORMALIZATION = 1
    INVERSE_NORMALIZATION = 2


class Scaler:
    def __init__(self, args):
        self.normalize_target = args.normalize_target  # 归一化的目标
        self.normalize_method = args.normalize_method  # 归一化的方法
        self.save_path = args.model_dir  # 保存归一化器的地址
        self.normalizers: dict = {}
        self.device = args.device

    @torch.no_grad()
    def scale(
        self,
        data: Tensor,
        data_type: str,
        is_train: bool,
        mode: ScalerMode = ScalerMode.NORMALIZATION,
    ) -> Tensor:
        """归一化数据

        Args:
            data(Tensor): 数据
            data_type(str): 数据类型 (x, y, none)
            is_train(bool): 是否是训练集
            mode(ScalerMode): 工作模式

        Returns:
            data(Tensor): 归一化后的数据
        """
        if data_type not in self.normalize_target:
            # 若输入数据不在归一化的目标中，则不进行归一化
            return data

        if self.normalize_method == "none":
            # 若归一化方法为 none，则不进行归一化
            return data  # do nothing

        if mode == ScalerMode.NORMALIZATION:
            data = self.normalize(
                data, data_type, self.save_path, is_train, self.normalize_method
            )
        elif mode == ScalerMode.INVERSE_NORMALIZATION:
            data = torch.Tensor(self.inverse_normalize(data, data_type))
        else:
            raise ValueError(
                f"工作模式未知，mode={mode}，合法的值为：ScalerMode.NORMALIZATION、"
                "ScalerMode.INVERSE_NORMALIZATION"
            )
        return data

    def inverse_normalize(self, data: Tensor, prefix) -> Tensor:
        """

        Args:
            data(Tensor): 数据
            prefix(str): 归一化器的前缀 [x, y, none]

        Returns:
            data(Tensor): 反归一化后的数据
        """
        # 从归一化后的数据恢复到原始数据
        if self.normalizers is None:
            raise FileNotFoundError("无归一化器，反归一化失败！")
        if prefix not in self.normalizers:
            raise ValueError(
                "未找到目标归一化器，反归一化失败！",
                f"normalizer_prefix={prefix}",
            )

        normalizer = self.normalizers[prefix]
        data = normalizer.inverse_transform(data.detach())
        return data

    def normalize(
        self, data: Tensor, prefix: str, save_path: str, is_train: bool, method: str
    ) -> Tensor:
        """
        归一化数据
        Args:
            data(Tensor): 数据
            prefix(str): 归一化器的前缀
            save_path(str): 保存归一化器的地址
            is_train(bool): 是否是训练集
            method(str): 归一化方法

        Returns:
            data(Tensor): 归一化后的数据
        """
        match method:
            case "zscore":
                normalizer = StandardScalerTorch()
            case _:
                raise ValueError(f"归一化方法未知，normalize_method={method}")

        if os.path.exists(os.path.join(save_path, f"{prefix}_normalizer.pkl")):
            # 检查地址下是否已有归一化器，若有则直接使用。
            with open(os.path.join(save_path, f"{prefix}_normalizer.pkl"), "rb") as f:
                _normalizer = pickle.load(f)
                if type(_normalizer) is not type(normalizer):
                    raise TypeError("归一化器类型错误！")
                self.normalizers[prefix] = _normalizer
            data = _normalizer.transform(data)  # 使用现有归一化器
        elif is_train:
            # 若无，则使用训练集的数据训练一个归一化器，并保存。
            data = normalizer.fit_transform(data)  # 更新 df 的值
            # 保存归一化器
            with open(os.path.join(save_path, f"{prefix}_normalizer.pkl"), "wb") as f:
                pickle.dump(normalizer, f)
            self.normalizers[prefix] = normalizer
        else:
            # 若无，且是测试集，则报错。
            raise FileNotFoundError("未找到归一化器，请先训练模型！")

        return data.to(self.device)
