"""
* 现代的神经网络结构有一些常用的小组件。
* 把它们都放在这个单独的文件 submodules.py 中，可以在各种任务中复用。
*
* File: submodules.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-19 03:04:08
* ----------------------------
* Modified: 2024-06-02 15:30:45
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
*
* 2024-06-02 12:25:52	FK	更新 DNN
"""

from typing import Optional

import torchinfo
from torch import nn


class DNN(nn.Module):
    """深度神经网络 (DNN) 模块。又名多层感知器（MLP）、全连接层（FC）、稠密（Dense）层。

    该模块在 torch 中由一系列 Linear 层组成，可选择批处理归一化、激活函数和剔除正则化。

    Raises:
        ValueError: 如果 width 为空列表。
        ValueError: 如果 width 的长度小于 2。
        ValueError: 如果 width 中有非正整数的元素。

    Attributes:
        dropout (Optional[nn.Dropout]): Dropout 层，如果 dropout_rate 不为 None，
            则为 Dropout 层，否则为 None。
        use_bn (bool): 是否使用批处理归一化。
        activation (Optional[str]): 激活函数的名称。
        depth (int): 线性层的数量。
        linears (nn.ModuleList): 由一系列 Linear 层组成的模块列表。
        bn (Optional[nn.BatchNorm1d]): 由一系列 BatchNorm1d 层组成的模块列表，
            如果 use_bn 为 True，则为 BatchNorm1d 层列表，否则为 None。
        activation_layers (Optional[nn.ModuleList]): 由一系列激活函数层组成的模块列表，
            如果 activation 不为 None，则为激活函数层列表，否则为 None。

    Methods:
        __init__: 初始化函数。
        forward: 前向传播函数。
    """

    def __init__(
        self,
        width: list[int],
        activation: Optional[str] = None,
        dropout_rate: Optional[float] = None,
        use_bn: bool = False,
    ):
        """
        Args:
            width (list[int]): 一个整数列表，表示每个线性层的输入和输出维度。
            activation (Optional[str]): 激活函数的名称。默认为 None。
            dropout_rate (Optional[float]): Dropout 正则化的比率。默认为 None。
            use_bn (Optional[bool]): 是否使用批处理归一化。默认为 False。
        Examples:
            >>> dnn = DNN([2, 3, 4, 5], activation="relu", dropout_rate=0.1, use_bn=True)
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None
        self.use_bn = use_bn
        self.activation = activation
        self.depth = len(width) - 1

        if not width:
            raise ValueError("hidden_units is empty!!")
        if self.depth < 1:
            raise ValueError("hidden_units must be a list of length greater than 0")
        if not all((x > 0 for x in width)):
            raise ValueError("hidden_units must be a list of positive integers")

        self.linears = nn.ModuleList(
            [nn.Linear(width[i], width[i + 1]) for i in range(self.depth)]
        )

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(width[i + 1]) for i in range(self.depth - 1)]
            )

        if self.activation is not None:
            self.activation_layers = nn.ModuleList(
                [
                    activation_layer(activation, width[i + 1])
                    for i in range(self.depth - 1)
                ]
            )

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):
            _layer_out = self.linears[i](deep_input)

            if i == self.depth - 1:
                # 做为通用模块，最后一层不需要激活函数
                return _layer_out

            if self.use_bn:
                _layer_out = self.bn[i](_layer_out)
            _layer_out = self.activation_layers[i](_layer_out)
            if self.dropout:
                _layer_out = self.dropout(_layer_out)

            deep_input = _layer_out
        return deep_input


class Identity(nn.Module):
    """
    A simple identity module that returns the input as it is.
    """

    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


def activation_layer(act_name, hidden_size=None):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation
    Return:
        act_layer: activation layer
    """
    if act_name is None:
        act_layer = Identity()
    elif issubclass(act_name.__class__, nn.Module):
        act_layer = act_name()
    elif isinstance(act_name, str):
        match act_name.lower():
            case "sigmoid":
                act_layer = nn.Sigmoid()
            case "relu":
                act_layer = nn.ReLU(inplace=True)
            case "prelu":
                act_layer = nn.PReLU()
            case "leakyrelu":
                act_layer = nn.LeakyReLU(0.01)
            case "tanh":
                act_layer = nn.Tanh()
            case "softmax":
                act_layer = nn.Softmax(dim=1)
            case "identity", "linear":
                act_layer = Identity()
            case _:
                raise NotImplementedError(act_name)
    else:
        raise NotImplementedError

    return act_layer


if __name__ == "__main__":
    dnn = DNN([2, 3, 4, 5], activation="relu", dropout_rate=0.1, use_bn=True)
    print(dnn)
    torchinfo.summary(dnn, (8, 2))
