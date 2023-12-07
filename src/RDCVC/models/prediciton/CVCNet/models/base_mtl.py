"""
* 基础的 MTL 模块，兼容单任务模型、多任务模型（底层共享 + 输出层分割）、
* 多任务模型（底层共享 + 任务特定塔）
*
* File: shared_bottom.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-28 11:15:19
* ----------------------------
* Modified: 2023-12-05 12:53:54
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

from typing import Dict, List

import torch

from .submodules import DNN


class BaseMTL(torch.nn.Module):
    """base multi-task model with shared bottom layer"""

    def __init__(
        self,
        inputs_dim: int,
        target_dict: Dict[str, int],
        bottom_units: List[int],
        tower_units: List[int] | None,
        activation: str = "leakyrelu:0.1",
    ):
        super().__init__()
        self.target_dict = target_dict
        self.inputs_dim = inputs_dim
        self.hidden_units = bottom_units

        self.bottom = DNN(
            inputs_dim=inputs_dim,
            hidden_units=bottom_units,
            activation=activation,
        )

        if tower_units is not None:
            self.towers = torch.nn.ModuleDict(
                {
                    f"{name}": DNN(
                        inputs_dim=bottom_units[-1],
                        hidden_units=tower_units,
                        activation=activation,
                    )
                    for name in target_dict
                }
            )
            assert len(self.target_dict) == len(self.towers)
        else:
            # the Identity layer as a placeholder avoids
            # having to determine if there is a tower in the forward.
            self.towers = torch.nn.ModuleDict(
                {f"{name}": torch.nn.Identity() for name in target_dict}
            )
        self.task_dense = torch.nn.ModuleDict(
            {
                f"{name}": DNN(
                    inputs_dim=tower_units[-1] if tower_units else bottom_units[-1],
                    hidden_units=[target_dict[name]],
                    activation=activation,
                )
                for name in target_dict
            }
        )

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        bottom_out = self.bottom(inputs)
        for name in self.target_dict.keys():
            tower_out = self.towers[f"{name}"](bottom_out)
            task_out = self.task_dense[f"{name}"](tower_out)
            outputs.append(task_out)

        return outputs
