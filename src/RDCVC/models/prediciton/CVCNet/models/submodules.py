"""
* 现代的神经网络结构有一些常用的小组件，比如 conv-bn-relu 这样的结构。
* 把它们都放在这个单独的文件 submodules.py 中，可以在各种任务中复用。
*
* File: submodules.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-19 03:04:08
* ----------------------------
* Modified: 2023-11-29 04:45:42
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

from typing import Optional

import torch
import torch.nn as nn


class DNN(nn.Module):
    """Deep Neural Network (DNN) module. AKA The Multi Layer Perceptron (MLP)

    This module implements a feedforward neural network with customizable architecture.
    It consists of a series of linear layers with optional batch normalization,
    activation functions, and dropout regularization.

    Input
      - nD tensor with shape: ``(batch_size, ..., input_dim)``.
        The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

    Output
      - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``.
        For instance, for a 2D input with shape ``(batch_size, input_dim)``,
        the output would have shape ``(batch_size, hidden_size[-1])``.

    Parameters:
        inputs_dim (int): Dimensionality of the input features.
        hidden_units (list): list of positive integer, the layer number and units in each layer.
        activation (str, optional): Activation function to be applied between hidden layers.
            Defaults to "relu".
        l2_reg (float, optional): L2 regularization strength. float between 0 and 1.
            Defaults to 0.0.
        dropout_rate (float, optional): Dropout rate to be applied after each hidden layer.
            float in [0,1). Defaults to 0.0.
        use_bn (bool, optional): Flag indicating whether to use batch normalization.
            Defaults to False.
        init_std (float, optional): Standard deviation of the normal distribution used for weight initialization.
            Defaults to 0.0001.

    Attributes:
        dropout_rate (float): Dropout rate.
        dropout (nn.Dropout): Dropout layer.
        l2_reg (float): L2 regularization strength.
        use_bn (bool): Flag indicating whether batch normalization is used.
        linears (nn.ModuleList): List of linear layers.
        bn (nn.ModuleList): List of batch normalization layers (if use_bn is True).
        activation_layers (nn.ModuleList): List of activation layers.
    """

    def __init__(
        self,
        inputs_dim: int,
        hidden_units: list[int],
        activation: Optional[str] = None,
        dropout_rate: Optional[float] = None,
        use_bn: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else None
        self.use_bn = use_bn
        self.activation = activation

        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [
                nn.Linear(hidden_units[i], hidden_units[i + 1])
                for i in range(len(hidden_units) - 1)
            ]
        )

        if self.use_bn:
            self.bn = nn.ModuleList(
                [
                    nn.BatchNorm1d(hidden_units[i + 1])
                    for i in range(len(hidden_units) - 1)
                ]
            )

        # t
        if self.activation is not None:
            self.activation_layers = nn.ModuleList(
                [
                    activation_layer(activation, hidden_units[i + 1])
                    for i in range(len(hidden_units) - 1)
                ]
            )

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            if self.dropout:
                fc = self.dropout(fc)
            deep_input = fc
        return deep_input


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
    elif isinstance(act_name, str):
        if act_name.lower() == "sigmoid":
            act_layer = nn.Sigmoid()
        elif act_name.lower() == "linear":
            act_layer = Identity()
        elif act_name.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif act_name.split(":")[0].lower() == "leakyrelu":
            act_layer = nn.LeakyReLU(float(act_name.split(":")[1]))
        elif act_name.lower() == "prelu":
            act_layer = nn.PReLU()
        elif act_name.split(":")[0].lower() == "softmax":
            act_layer = nn.Softmax(dim=int(act_name.split(":")[1]))
        else:
            raise NotImplementedError(act_name)

    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer
