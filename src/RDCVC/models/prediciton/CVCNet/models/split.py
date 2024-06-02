"""
*
*
* File: split.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-28 11:15:57
* ----------------------------
* Modified: 2024-06-02 13:15:57
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

from typing import Dict, List

import torch

from .submodules import DNN


class SplitMTL(torch.nn.Module):
    """共享底层的基础多任务模型，仅在最后一层按任务分割

    Attributes:
        width (List[int]): 模型的宽度
        target_dict (Dict[str, int]): 任务字典
        activation (str): 激活函数

    Methods:
        forward(inputs: torch.Tensor) -> List[torch.Tensor]: 前向传播

    Example:
        >>> model = SplitMTL(
        >>>     width=[18, 32, 32, 32],
        >>>     target_dict={"Airflow": 4, "Pres": 6},
        >>>     activation="leakyrelu",
        >>> )
    """

    def __init__(
        self,
        width: List[int],
        target_dict: Dict[str, int],
        activation: str = "leakyrelu",
    ):
        """
        Args:
            width (List[int]): 模型的宽度，包括输入层
            target_dict (Dict[str, int]): 任务字典，任务名 -> 任务维度
            activation (str): 激活函数
        """
        super().__init__()
        self.target_dict = target_dict

        self.bottom = DNN(
            width=width,
            activation=activation,
        )

        self.task_dense = torch.nn.ModuleDict(
            {
                f"{name}": DNN(
                    width=[width[-1], target_dict[name]],
                    activation=activation,
                )
                for name in target_dict
            }
        )

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        _bottom_out = self.bottom(inputs)

        outputs = []
        for name in self.target_dict.keys():
            task_out = self.task_dense[f"{name}"](_bottom_out)
            outputs.append(task_out)

        return outputs


if __name__ == "__main__":
    # test SplitMTL
    model = SplitMTL(
        width=[18, 32, 32, 32],
        target_dict={"Airflow": 4, "Pres": 6},
        activation="leakyrelu",
    )

    print(model)
    import torchinfo

    torchinfo.summary(model, (3, 18))
    print((model(torch.randn(3, 18).to("cuda"))))
    print("SplitMTL test passed.")
