"""
* 基础的 MTL 模块，兼容单任务模型、多任务模型（底层共享 + 输出层分割）、
* 多任务模型（底层共享 + 任务特定塔）
*
* File: split.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-28 11:15:57
* ----------------------------
* Modified: 2024-06-02 15:35:40
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

from typing import Dict, List

import torch

# from src.RDCVC.models.prediciton.CVCNet.models.submodules import DNN
from .submodules import DNN


class BaseMTL(torch.nn.Module):
    """共享底层的基础多任务模型。

    Attributes:
        bottom (DNN): 共享底层
        task_tower (torch.nn.ModuleDict): 任务特定塔
        task_outlayer (torch.nn.ModuleDict): 任务输出层

    Methods:
        forward(inputs: torch.Tensor) -> List[torch.Tensor]: 前向传播

    Example:
        >>> model = SplitMTL(
        >>>     bottom_width=[18, 32],
        >>>     tower_width=[32, 64],
        >>>     target_dict={"Airflow": 4, "Pres": 6},
        >>>     activation="leakyrelu",
        >>> )
    """

    def __init__(
        self,
        bottom_width: List[int],
        tower_width: List[int],
        target_dict: Dict[str, int],
        activation: str = "leakyrelu",
    ):
        """
        Args:
            bottom_width (List[int]): 模型共享底层的宽度
            tower_width (List[int]): 模型任务特定塔的宽度，如果为空列表则不使用任务特定塔
            target_dict (Dict[str, int]): 任务字典，任务名 -> 任务维度
            activation (str): 激活函数

        """
        super().__init__()
        self.target_dict = target_dict

        # -------------------------------------------------------- #
        self.bottom = DNN(
            width=bottom_width,
            activation=activation,
        )

        if len(tower_width) == 0:
            self.task_tower = torch.nn.ModuleDict(
                {f"{name}": torch.nn.Identity() for name in target_dict}
            )
        else:
            self.task_tower = torch.nn.ModuleDict(
                {
                    f"{name}": DNN(
                        width=[bottom_width[-1]] + tower_width,
                        activation=activation,
                    )
                    for name in target_dict
                }
            )

        self.task_outlayer = torch.nn.ModuleDict(
            {
                f"{name}": DNN(
                    width=[
                        tower_width[-1] if tower_width else bottom_width[-1],
                        target_dict[name],
                    ],
                    # activation=activation, # 输出层暂时不使用激活函数
                )
                for name in target_dict
            }
        )

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        _bottom_out = self.bottom(inputs)

        outputs = []
        for name in self.target_dict.keys():
            tower_out = self.task_tower[f"{name}"](_bottom_out)
            task_out = self.task_outlayer[f"{name}"](tower_out)
            outputs.append(task_out)

        return outputs


if __name__ == "__main__":
    model = BaseMTL(
        bottom_width=[18, 32],
        tower_width=[],
        target_dict={"Airflow": 4, "Pres": 6},
        activation="leakyrelu",
    )

    print(model)
    import torchinfo

    torchinfo.summary(model, (3, 18))
    print((model(torch.randn(3, 18).to("cuda"))))
    print("SplitMTL test passed.")
