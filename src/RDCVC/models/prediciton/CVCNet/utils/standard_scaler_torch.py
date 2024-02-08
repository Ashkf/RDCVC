"""
* Standardized scaler implemented in pytorch
*
* File: standard_scaler_torch.py
* Author: Fan Kai
* Soochow University
* Created: 2024-02-07 00:20:50
* ----------------------------
* Modified: 2024-02-09 00:04:07
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
*
* 2024-02-08 23:59:4	FK	complete
"""

import torch
from torch import Tensor


class StandardScalerTorch:
    """通过减去平均值并缩放至单位方差来标准化特征。

    样本 `x` 的 zscore 计算公式为：

        z = (x - u) / s

    其中 `u` 是训练样本的平均值，`s` 是训练样本的标准偏差。

    通过计算训练集中样本的相关统计数据，对每个特征进行独立的居中和缩放。
    然后存储平均值和标准偏差，以便在以后的数据中使用

    Attributes:
        mean_ (Tensor): 每个特征的平均值。
        var_ (Tensor): 每个特征的方差。
        scale_ (Tensor): 每个特征的标准差。
        n_features_in_ (int): 输入特征的数量。
        n_samples_seen_ (int): 观察到的样本数量。
    """

    def __init__(self):
        pass

    def _reset(self):
        """重置放缩器的内部数据，不涉及__init__中的参数。"""
        if hasattr(self, "scale_"):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X: Tensor):
        """Compute the mean and std to be used for later scaling.

        Args:
            X (Tensor): 用于计算平均值和标准差的数据，
                以备沿特征轴进行缩放。shape: (n_samples, n_features)

        Returns:
            self: 拟合后的归一化器。
        """
        self._reset()

        if not isinstance(X, Tensor):
            raise ValueError("类 StandardScaler_torch 仅支持 torch.Tensor 类型的输入")

        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        self.mean_ = X.mean(dim=0)
        self.var_ = X.var(dim=0, correction=False)
        self.scale_ = X.std(dim=0, correction=False)
        return self

    def transform(self, X: Tensor) -> Tensor:
        """通过居中和缩放进行标准化。

        Args:
            X (Tensor): 用于沿特征轴缩放的数据。
                shape: (n_samples, n_features)

        Returns:
            Tensor: 缩放后的数据。shape: (n_samples, n_features)
        """
        if not hasattr(self, "scale_"):
            raise ValueError("请先拟合数据，然后再进行转换")

        # 除 0 检查
        if (self.scale_ == 0).any():
            raise ValueError("标准差为 0 时，无法进行缩放")

        return (X - self.mean_) / torch.max(self.scale_, torch.tensor(1e-7))

    def inverse_transform(self, X: Tensor) -> Tensor:
        """将数据缩放回原始表征。

        Args:
            X (Tensor): 用于沿特征轴缩放的数据。
                shape: (n_samples, n_features)

        Returns:
            Tensor: 缩放后的数据。shape: (n_samples, n_features)
        """
        if not hasattr(self, "scale_"):
            raise ValueError("请先拟合数据，然后再进行转换")

        return X * self.scale_ + self.mean_

    def fit_transform(self, X: Tensor) -> Tensor:
        """先拟合数据，然后转换它。

        Args:
            X (Tensor): 用于计算平均值和标准差的数据，
                shape: (n_samples, n_features)

        Returns:
            Tensor: 缩放后的数据。shape: (n_samples, n_features)
        """
        self.fit(X)
        return self.transform(X)
