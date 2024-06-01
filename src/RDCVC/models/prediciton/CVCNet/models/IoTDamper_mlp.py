"""
*
*
* File: IoTDamper_mlp.py
* Author: Fan Kai
* Soochow University
* Created: 2024-05-31 11:34:00
* ----------------------------
* Modified: 2024-05-31 16:44:31
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

from torch import nn


class MLPL4VPTDs(nn.Module):
    """4 层全连接神经网络，输入为风量 + 压差，输出为 2 个阀片开度"""

    def __init__(self):
        # 输入（2 个）：风量 + 压差
        # 输出（2 个）：2 个阀片开度
        super().__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, 2)

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class MLPL12VPTDs(nn.Module):
    """12 层全连接神经网络，输入为风量 + 压差，输出为 2 个阀片开度"""

    def __init__(self):
        # 输入（2 个）：风量 + 压差
        # 输出（2 个）：2 个阀片开度
        super().__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, 8)
        self.layer4 = nn.Linear(8, 8)
        self.layer5 = nn.Linear(8, 8)
        self.layer6 = nn.Linear(8, 8)
        self.layer7 = nn.Linear(8, 8)
        self.layer8 = nn.Linear(8, 8)
        self.layer9 = nn.Linear(8, 8)
        self.layer10 = nn.Linear(8, 8)
        self.layer11 = nn.Linear(8, 8)
        self.layer12 = nn.Linear(8, 2)

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        x = nn.functional.relu(self.layer4(x))
        x = nn.functional.relu(self.layer5(x))
        x = nn.functional.relu(self.layer6(x))
        x = nn.functional.relu(self.layer7(x))
        x = nn.functional.relu(self.layer8(x))
        x = nn.functional.relu(self.layer9(x))
        x = nn.functional.relu(self.layer10(x))
        x = nn.functional.relu(self.layer11(x))
        x = self.layer12(x)
        return x


class DAPN4(nn.Module):
    """Damper Airflow Prediction Network

    4 层全连接神经网络，输入为 2 个阀片开度 + 压差，输出为风量
    """

    def __init__(self):
        # 输入（3 个）：2 个阀片开度 + 压差
        # 输出（1 个）：风量
        super().__init__()
        self.layer1 = nn.Linear(3, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, 1)

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class DAPN12(nn.Module):
    """Damper Airflow Prediction Network

    12 层全连接神经网络，输入为 2 个阀片开度 + 压差，输出为风量
    """

    def __init__(self):
        # 输入（3 个）：2 个阀片开度 + 压差
        # 输出（1 个）：风量
        super().__init__()
        self.layer1 = nn.Linear(3, 200)
        self.layer2 = nn.Linear(200, 200)
        self.layer3 = nn.Linear(200, 200)
        self.layer4 = nn.Linear(200, 200)
        self.layer5 = nn.Linear(200, 200)
        self.layer6 = nn.Linear(200, 200)
        self.layer7 = nn.Linear(200, 200)
        self.layer8 = nn.Linear(200, 200)
        self.layer9 = nn.Linear(200, 200)
        self.layer10 = nn.Linear(200, 200)
        self.layer11 = nn.Linear(200, 200)
        self.layer12 = nn.Linear(200, 1)

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        x = nn.functional.relu(self.layer4(x))
        x = nn.functional.relu(self.layer5(x))
        x = nn.functional.relu(self.layer6(x))
        x = nn.functional.relu(self.layer7(x))
        x = nn.functional.relu(self.layer8(x))
        x = nn.functional.relu(self.layer9(x))
        x = nn.functional.relu(self.layer10(x))
        x = nn.functional.relu(self.layer11(x))
        x = self.layer12(x)
        return x
