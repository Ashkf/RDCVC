"""
*
*
* File: split.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-28 11:15:57
* ----------------------------
* Modified: 2023-12-05 12:58:57
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

from typing import Dict, List

import torch

from .submodules import DNN


class SplitMTL(torch.nn.Module):
    """base multi-task model with shared bottom layer

    Args:
        inputs_dim (int): dimension of input
        target_dict (Dict[str, int]): dict of task name and output dimension
        bottom_units (List[int]): units of bottom layer
        activation (_type_, optional): activation function.
            Defaults to "leakyrelu:0.1".
    """

    def __init__(
        self,
        inputs_dim: int,
        target_dict: Dict[str, int],
        bottom_units: List[int],
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

        # the Identity layer as a placeholder avoids
        # having to determine if there is a tower in the forward.
        self.towers = torch.nn.ModuleDict(
            {f"{name}": torch.nn.Identity() for name in target_dict}
        )

        self.task_dense = torch.nn.ModuleDict(
            {
                f"{name}": DNN(
                    inputs_dim=bottom_units[-1],
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
            task_out = self.task_dense[f"{name}"](bottom_out)
            outputs.append(task_out)

        return outputs
