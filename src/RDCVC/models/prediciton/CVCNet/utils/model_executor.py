"""
ModelExecutor 类

ModelExecutor 类是一个模型执行器，用于执行模型的 forward。
"""

import torch
from torch import Tensor


class ModelExecutor:
    def __init__(self, model_type, model):
        if not issubclass(model.__class__, torch.nn.Module):
            raise TypeError("model must be a subclass of torch.nn.Module")
        self.model_type = model_type
        self.model = model

    def forward(
        self,
        item,
        label,
    ) -> (dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]):
        """执行一次 forward

        Args:
            item: input data
            label: ground truth
        Returns:
            pred(dict[str, Tensor]): model output
        """
        return (
            self.forward_mtl(item, label)
            if "mtl" in self.model_type.split("_")[0]
            else self.forward_normal(item, label)
        )

    def forward_mtl(
        self, item: dict[str, Tensor], label: dict[str, Tensor]
    ) -> dict[str, Tensor]:  # sourcery skip: remove-dict-keys
        """mtl 的前向传播函数

        Args:
            item: input data
            label: ground truth
        Returns:
            pred: model output
        """
        # Move the tensor to GPU
        # label = label.to('cuda')
        _input = torch.stack([item[k] for k in item.keys()], dim=1)
        _pred_heads = self.model(_input)  # 调用模型进行前向传播
        _pred = torch.cat(_pred_heads, dim=1)  # 横向拼接
        pred: dict[str, Tensor] = {
            k: _pred[:, i] for i, k in enumerate(label.keys())
        }  # 格式化输出
        return pred

    def forward_normal(self, item: dict[str, Tensor], label: dict[str, Tensor]):
        # sourcery skip: remove-dict-keys
        """一般的前向传播函数"""
        _input = torch.stack([item[k] for k in item.keys()], dim=1)
        _pred = self.model(_input)  # 调用模型进行前向传播
        pred: dict[str, Tensor] = {
            k: _pred[:, i] for i, k in enumerate(label.keys())
        }  # 格式化输出
        return pred
