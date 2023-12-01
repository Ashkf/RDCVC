"""
*
*
* File: shared_bottom.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-28 11:15:19
* ----------------------------
* Modified: 2023-11-28 11:47:49
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

from collections import OrderedDict
from typing import Dict, List

import torch

from .submodules import DNN


# TODO:TEST
class shared_bottom:
    """base multi-task model

    shared bottom and task-specific towers
    or no task-specific towers
    """

    def __init__(
        self,
        inputs_dim: int,
        target_dict: Dict[str, int],
        bottom_units: List[int],
        tower_units: List[int],
        activation: str = "leakyrelu:0.1",
    ):
        self.target_dict = target_dict
        self.inputs_dim = inputs_dim
        self.hidden_units = bottom_units
        self.bottom = DNN(
            inputs_dim=inputs_dim,
            hidden_units=bottom_units,
            activation=activation,
        )
        self.towers = torch.nn.ModuleDict(
            DNN(
                inputs_dim=bottom_units[-1],
                hidden_units=tower_units,
                activation=activation,
            )
        )
        self.task_dense = torch.nn.ModuleDict(
            [
                torch.nn.Linear(tower_units[-1], target_dict[name], bias=False)
                for name in target_dict
            ]
        )
        assert len(self.target_dict) == len(self.towers)

    def forward(self, inputs):
        outputs = OrderedDict()
        bottom_out = self.bottom(inputs)
        for id, name in enumerate(self.target_dict):
            tower_out = self.towers[id](bottom_out)
            task_out = self.task_dense[id](tower_out)
            outputs[name] = task_out

        return outputs
