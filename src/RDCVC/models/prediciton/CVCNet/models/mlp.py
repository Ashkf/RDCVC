"""
*
*
* File: mlp.py
* Author: Fan Kai
* Soochow University
* Created: 2024-05-31 12:06:39
* ----------------------------
* Modified: 2024-05-31 12:59:19
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import torchinfo
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, width: list[int], activation_fn="leakyrelu0.01"):
        """多层感知机

        包含输入层、隐藏层、输出层，隐藏层使用 LeakyReLU 激活函数

        Args:
            activation_fn: 激活函数
        """
        super().__init__()

        num_layers = len(width) - 1
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
                "MLP's activation_fn must be in "
                "['leakyrelu0.01', 'relu', 'sigmoid', 'tanh']"
            )

        self.layers = nn.ModuleList()

        # 其他隐藏层
        for i in range(num_layers):
            self.layers.append(nn.Linear(width[i], width[i + 1]))
            self.layers.append(activation_fn)

    def forward(self, x) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    mlp = MLP([2, 4, 4, 1])
    print(mlp)
    torchinfo.summary(mlp, (2,))
