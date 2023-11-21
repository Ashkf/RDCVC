"""
现代的神经网络结构有一些常用的小组件，比如 conv-bn-relu 这样的结构。
把它们都放在这个单独的文件 submodules.py 中，可以在各种任务中复用。
"""

import torch
import torch.nn as nn


def conv2d_bn_relu(
    in_dim, out_dim, kernel, stride=1, pad=0, dilate=1, group=1, bias=True
):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel, stride, pad, dilate, group),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
    )


def up_conv2d(in_dim, out_dim, kernel=3, pad=1, up_scale=2):
    return nn.Sequential(
        nn.Upsample(scale_factor=up_scale, mode="nearest"),
        nn.Conv2d(in_dim, out_dim, kernel, padding=pad),
    )


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation_fn="leakyrelu0.01",
    ):
        """多层感知机

        包含输入层、隐藏层、输出层，隐藏层使用 LeakyReLU 激活函数

        Args:
            in_dim: 输入维度
            out_dim: 输出维度
            hidden_dim: 隐藏层维度
            num_layers: 隐藏层数量
            activation_fn: 激活函数
        """
        super(MLPBlock, self).__init__()
        if num_layers <= 2:
            raise ValueError("MLPBlock's num_layers must be greater than 2")

        if activation_fn == "leakyrelu0.01":
            activation_fn = nn.LeakyReLU(0.01)
        elif activation_fn == "relu":
            activation_fn = nn.ReLU()
        elif activation_fn == "sigmoid":
            activation_fn = nn.Sigmoid()
        elif activation_fn == "tanh":
            activation_fn = nn.Tanh()
        else:
            raise ValueError(
                "MLPBlock's activation_fn must be in "
                "['leakyrelu0.01', 'relu', 'sigmoid', 'tanh']"
            )

        self.layers = nn.ModuleList()

        # 输入层->隐藏层 1
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.layers.append(activation_fn)
        # 其他隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(activation_fn)
        # 隐藏层 n->输出层
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
