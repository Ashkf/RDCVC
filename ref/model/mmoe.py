"""
*
*
* File: mmoe.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-28 06:29:14
* ----------------------------
* Modified: 2023-11-30 02:48:12
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

from typing import Dict, List, Optional, Union

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


class MMoE(torch.nn.Module):
    """
    single layer MMoE
    """

    def __init__(
        self,
        inputs_dim: int,
        target_dict: Dict[str, int],
        num_experts: int,
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
        """
        Initializes the MMOE model.

        Args:
            inputs_dim (int): The dimension of the input features.
            target_dict (Dict[str, int]):
                A dictionary mapping target names to their corresponding output dimensions.
            num_experts (int):
                The number of experts in the MMOE model.
            expert_units (List[int]):
                A list of integers representing the number of units in each expert layer.
                e.g. [512, 256, 128]
            expert_activation (str):
                The activation function to be used in the expert layers.
            expert_dropout_rate (Optional[float]):
                The dropout rate to be applied in the expert layers.
            expert_use_bn (bool):
                A flag indicating whether to use batch normalization in the expert layers.
            tower_units (Optional[List[int]]):
                A list of integers representing the number of units in each tower layer.
                Defaults to None.
            tower_activation (Optional[str]):
                The activation function to be used in the tower layers. Defaults to None.
            tower_dropout_rate (Optional[float]):
                The dropout rate to be applied in the tower layers. Defaults to None.
            tower_use_bn (Optional[bool]):
                A flag indicating whether to use batch normalization in the tower layers.
                Defaults to False.
            is_last_layer (bool):
                A flag indicating whether this is the last layer of the MMOE model.
                Defaults to True.
        """
        super().__init__()
        num_task = len(target_dict)
        self.num_experts = num_experts
        self.target_dict = target_dict
        self.is_last_layer = is_last_layer

        # ------------------------ experts ----------------------- #
        # per expert:
        #   input: x, represents, (batch_size, inputs_dim)
        #   output: y, represents, (batch_size, expert_units[-1])
        self.experts = torch.nn.ModuleList(
            [
                Expert(
                    inputs_dim,
                    expert_units,
                    expert_activation,
                    expert_dropout_rate,
                    expert_use_bn,
                )
                for _ in range(num_experts)
            ]
        )

        # ------------------------- gates ------------------------ #
        # per gate:
        #   input: x, represents, (batch_size, inputs_dim)
        #         or output of previous layer, (batch_size, expert_units[-1])
        #   output: y, represents, (batch_size, num_experts)
        #
        # num of gates:
        #   如果是最后一层，那么 gate 的数量应该是 task 的数量
        #   其他层的话，gate 的数量一般等于 experts 的数量
        self.num_gate = num_task if is_last_layer else num_experts
        self.gates = torch.nn.ModuleList(
            [
                DNN(
                    inputs_dim,
                    [num_experts],
                    activation="softmax:1",
                )
                for _ in range(self.num_gate)
            ]
        )

        if is_last_layer:
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
                    for _ in range(num_task)
                ]
            )
            # ------------------------ output ------------------------ #
            self.task_dense = torch.nn.ModuleList(
                [
                    torch.nn.Linear(tower_units[-1], target_dict[name])
                    for name in target_dict
                ]
            )

    def forward(
        self, inputs: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        Returns:
            Union[Dict[str, torch.Tensor], List[torch.Tensor]]:
                if as last layer, return a dict of outputs for each task.
                else, return a list of outputs for each gate.
        """
        # The first layer uses the raw input as a list element,
        # and the other layers use the gate outputs from the previous layer.
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs] * self.num_experts

        experts_outputs = []  # length: num_experts
        for idx, expert in enumerate(self.experts):
            # expert:
            # Input shape: (batch_size, inputs_dim)
            # Output shape: (batch_size, expert_units[-1])
            expert_output = expert(inputs[idx])
            experts_outputs.append(expert_output)

        #  (batch_size, expert_units[-1], num_experts)
        experts_outputs = torch.stack(experts_outputs, dim=-1)

        if not self.is_last_layer:
            outputs = []  # length: num_experts
            for idx, gate in enumerate(self.gates):
                # Input shape: (batch_size, inputs_dim)
                # Output shape: (batch_size, num_experts)
                gates_weight = gate(inputs[idx])
                gate_outputs = torch.einsum("ijk,ik->ij", experts_outputs, gates_weight)
                outputs.append(gate_outputs)

        if self.is_last_layer:
            outputs = {}  # length: num_task
            for idx, gate in enumerate(self.gates):
                # Input shape: (batch_size, inputs_dim)
                # Output shape: (batch_size, num_experts)
                gates_weight = gate(inputs[idx])
                gate_outputs = torch.einsum("ijk,ik->ij", experts_outputs, gates_weight)
                # towers:
                # Input shape: (batch_size, expert_units[-1])
                # Output shape: (batch_size, tower_units[-1])
                tower_outputs = self.towers[idx](gate_outputs)

                # output:
                # Input shape: (batch_size, tower_units[-1])
                # Output shape: (batch_size, target_dict[name])
                output = self.task_dense[idx](tower_outputs)

                outputs[list(self.target_dict.keys())[idx]] = output
        return outputs


class ML_MMoE(torch.nn.Module):
    """Multi-layer Multi-gate Mixture-of-experts model."""

    def __init__(
        self,
        inputs_dim: int,
        target_dict: Dict[str, int],
        num_layers: int,
        num_experts: int,
        expert_units: List[int],
        expert_activation: str,
        expert_dropout_rate: Optional[float],
        expert_use_bn: bool,
        tower_units: Optional[List[int]],
        tower_activation: Optional[str],
        tower_dropout_rate: Optional[float],
        tower_use_bn: Optional[bool],
    ):
        super().__init__()
        self.num_experts = num_experts
        self.mmoe_layers = torch.nn.ModuleList(
            [
                MMoE(
                    inputs_dim if layer_id == 0 else expert_units[-1],
                    target_dict,
                    num_experts,
                    expert_units,
                    expert_activation,
                    expert_dropout_rate,
                    expert_use_bn,
                    tower_units,
                    tower_activation,
                    tower_dropout_rate,
                    tower_use_bn,
                    is_last_layer=(layer_id == num_layers - 1),
                )
                for layer_id in range(num_layers)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        for layer in self.mmoe_layers:
            mmoe_outputs = layer(inputs)
            inputs = mmoe_outputs
        return mmoe_outputs
