"""
*
*
* File: ple.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-30 08:22:24
* ----------------------------
* Modified: 2023-11-30 02:35:25
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

from typing import Dict, List, Optional

import torch

from .submodules import DNN


class Expert(torch.nn.Module):
    def __init__(
        self,
        inputs_dim: int,
        hiden_units: List[int],
        activation: Optional[str] = "leakyrelu:0.1",
        dropout_rate: Optional[float] = None,
        use_bn: bool = False,
    ):
        super().__init__()
        self.kernel = DNN(
            inputs_dim,
            hiden_units,
            activation=activation,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
        )

    def forward(self, inputs):
        outputs = self.kernel(inputs)
        return outputs


class Tower(torch.nn.Module):
    def __init__(
        self,
        inputs_dim: int,
        hiden_units: List[int],
        activation: Optional[str] = "leakyrelu:0.1",
        dropout_rate: Optional[float] = None,
        use_bn: bool = False,
    ):
        super().__init__()
        self.kernel = DNN(
            inputs_dim,
            hiden_units,
            activation=activation,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
        )

    def forward(self, inputs):
        outputs = self.kernel(inputs)
        return outputs


class CGC(torch.nn.Module):
    """Customized Gate Control"""

    def __init__(
        self,
        inputs_dim,
        target_dict: Dict[str, int],
        num_tasks_experts,
        num_shared_experts,
        expert_units: List[int],
        expert_activation: str,
        expert_dropout_rate: Optional[float],
        expert_use_bn: bool,
        tower_units: Optional[List[int]] = None,
        tower_activation: Optional[str] = None,
        tower_dropout_rate: Optional[float] = None,
        tower_use_bn: Optional[bool] = False,
        is_last_layer: bool = True,
    ):
        """Initialize CGC."""
        super().__init__()
        self.num_task = len(target_dict)
        self.is_last_layer = is_last_layer

        # ------------------------ epxerts ----------------------- #
        self.shared_experts = torch.nn.ModuleList(
            [
                Expert(
                    inputs_dim,
                    expert_units,
                    expert_activation,
                    expert_dropout_rate,
                    expert_use_bn,
                )
                for _ in range(num_shared_experts)
            ]
        )

        self.tasks_experts = [
            torch.nn.ModuleList(
                [
                    Expert(
                        inputs_dim,
                        expert_units,
                        expert_activation,
                        expert_dropout_rate,
                        expert_use_bn,
                    )
                    for _ in range(num_tasks_experts)
                ]
            )
            for task in target_dict
        ]

        # ------------------------- gates ------------------------ #
        self.gates = {
            "shared": torch.nn.ModuleList(
                [
                    DNN(
                        inputs_dim,
                        [num_shared_experts + num_tasks_experts * self.num_task],
                        activation="softmax:1",
                    )
                ]
            )
        }
        self.gates.update(
            {
                task_name: torch.nn.ModuleList(
                    [
                        DNN(
                            inputs_dim,
                            [num_tasks_experts + num_shared_experts],
                            activation="softmax:1",
                        )
                    ]
                )
                for task_name in target_dict
            }
        )
        self.num_gates = self.num_task if self.is_last_layer else len(self.gates)
        if self.is_last_layer:
            # ------------------------ towers ------------------------ #
            self.towers = torch.nn.ModuleList(
                [
                    Tower(
                        expert_units[-1],
                        tower_units,
                        tower_activation,
                        tower_dropout_rate,
                        tower_use_bn,
                    )
                    for _ in range(self.num_task)
                ]
            )
            # ------------------------ output ------------------------ #
            self.task_dense = torch.nn.ModuleList(
                [
                    torch.nn.Linear(tower_units[-1], target_dict[name])
                    for name in target_dict
                ]
            )

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]: ...
