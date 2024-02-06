"""
* CVCNet: Cleanroom Ventilation Control neural Network
*
* File: cvcnet.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-26 02:48:55
* ----------------------------
* Modified: 2024-01-29 17:20:57
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

from typing import Dict, List, Optional, Tuple, Union

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


class ExtractionNet(torch.nn.Module):
    """
    Extraction Network Layer

    The Extraction Network is used to extract the features from the input data.
    The Extraction Network consists of a set of experts and a set of gates.

    Can be used as a single layer or a part of multi-layer network.
    """

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
        """
        Args:
            inputs_dim (int): The dimension of the input features.
            target_dict (Dict[str, int]): The dictionary of target names and their dimensions.
                e.g., {"PRESENCE": 6, "AIRFLOW": 4}
            num_tasks_experts (int): The number of task experts.
            num_shared_experts (int): The number of shared experts.
            expert_units (List[int]): The hidden units of experts. e.g., [5, 6, 7]
            expert_activation (str): The activation function of experts.
            expert_dropout_rate (Optional[float]): The dropout rate of experts.
            expert_use_bn (bool): Whether to use batch normalization in experts.
            tower_units (Optional[List[int]]): The hidden units of towers.
            tower_activation (Optional[str]): The activation function of towers.
            tower_dropout_rate (Optional[float]): The dropout rate of towers.
            tower_use_bn (Optional[bool]): Whether to use batch normalization in towers.
            is_last_layer (bool): Whether to use as the last layer.
        """
        super().__init__()
        self.target_dict = target_dict
        self.num_task = len(target_dict)
        self.num_gates = 4  # shared, SA, RA, EA
        self.is_last_layer = is_last_layer

        # ------------------------ Experts ----------------------- #
        # per expert:
        #   input: x, represents, (batch_size, inputs_dim)
        #   output: y, represents, (batch_size, expert_units[-1])
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
        self.SA_experts = torch.nn.ModuleList(
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
        self.RA_experts = torch.nn.ModuleList(
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
        self.EA_experts = torch.nn.ModuleList(
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

        # ------------------------- Gates ------------------------ #
        # per gate:
        #   input: x, represents, (batch_size, inputs_dim)
        #         or output of previous layer, (batch_size, expert_units[-1])
        #   output: y, represents, (batch_size, num_experts)
        # num of gates: 4, shared, SA, RA, EA
        self.shared_gate = DNN(
            inputs_dim,
            [num_shared_experts + num_tasks_experts * (self.num_gates - 1)],
            activation="softmax:1",
        )

        self.SA_gate = DNN(
            inputs_dim,
            [num_tasks_experts + num_shared_experts],
            activation="softmax:1",
        )

        self.RA_gate = DNN(
            inputs_dim,
            [num_tasks_experts + num_shared_experts],
            activation="softmax:1",
        )

        self.EA_gate = DNN(
            inputs_dim,
            [num_tasks_experts + num_shared_experts],
            activation="softmax:1",
        )

        if self.is_last_layer:
            # ------------------------ towers ------------------------ #
            # per tower:
            #   input: concated gated experts' out, (batch_size, expert_units[-1]*num_gates)
            #   output: y, represents, (batch_size, tower_units[-1])
            self.towers = torch.nn.ModuleList(
                [
                    Tower(
                        expert_units[-1] * self.num_gates,
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

    def forward(
        self, inputs: Union[torch.Tensor, List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Returns:
            Union[Dict[str, torch.Tensor], List[torch.Tensor]]:
                if as last layer, return a dict of outputs for each task.
                else, return a list of outputs for each gate.
        """

        if isinstance(inputs, torch.Tensor):
            inputs_shared = inputs_SA = inputs_RA = inputs_EA = inputs
        elif isinstance(inputs, list):
            assert len(inputs) == self.num_gates
            inputs_shared, inputs_SA, inputs_RA, inputs_EA = inputs

        # ------------------------ Experts ----------------------- #
        # expert:
        #   Input shape: (batch_size, inputs_dim)
        #   Output shape: (batch_size, expert_units[-1])
        # experts_outputs: [(batch_size, expert_units[-1]), ...)]
        shared_experts_outputs = [e(inputs_shared) for e in self.shared_experts]
        SA_experts_outputs = [e(inputs_SA) for e in self.SA_experts]
        RA_experts_outputs = [e(inputs_RA) for e in self.RA_experts]
        EA_experts_outputs = [e(inputs_EA) for e in self.EA_experts]

        # ------------------------- Gate ------------------------- #
        # num_experts are uncertain for each gate
        gate_shared_outputs = torch.einsum(
            "ijk,ik->ij",
            torch.stack(
                shared_experts_outputs
                + SA_experts_outputs
                + RA_experts_outputs
                + EA_experts_outputs,
                dim=-1,
            ),  # (batch_size, expert_units[-1]*num_gates, num_experts)
            self.shared_gate(inputs_shared),  # gate weights (batch_size, num_experts)
        )
        gate_SA_outputs = torch.einsum(
            "ijk,ik->ij",
            torch.stack(SA_experts_outputs + shared_experts_outputs, dim=-1),
            self.SA_gate(inputs_SA),  # gate weights (batch_size, num_experts)
        )
        gate_RA_outputs = torch.einsum(
            "ijk,ik->ij",
            torch.stack(RA_experts_outputs + shared_experts_outputs, dim=-1),
            self.RA_gate(inputs_RA),  # gate weights (batch_size, num_experts)
        )
        gate_EA_outputs = torch.einsum(
            "ijk,ik->ij",
            torch.stack(EA_experts_outputs + shared_experts_outputs, dim=-1),
            self.EA_gate(inputs_EA),  # gate weights (batch_size, num_experts)
        )

        # ------------------------ towers ------------------------ #
        # if as last layer, return a dict of outputs for each task.
        if self.is_last_layer:
            tower_in = torch.cat(
                (
                    gate_shared_outputs,
                    gate_SA_outputs,
                    gate_RA_outputs,
                    gate_EA_outputs,
                ),
                dim=1,
            )  # (batch_size, expert_units[-1]*num_gates)
            task_outs = []
            for tower, dense in zip(self.towers, self.task_dense, strict=True):
                task_outs.append(dense(tower(tower_in)))

            return task_outs

        return [gate_shared_outputs, gate_SA_outputs, gate_RA_outputs, gate_EA_outputs]


class CVCNet(torch.nn.Module):
    """
    Cleanroom Ventilation Control neural Network

    A deep multi-task learning based neural network for cleanroom ventilation control
    with inputs of 18 sensor data and outputs of 10 controlled variables.

    Each embedded submodule is a Deep Neural Network (DNN) i.e.,
    Multilayer Perceptron (MLP), consisting of multiple fully connected layers.
    """

    def __init__(
        self,
        inputs_dim: int,
        target_dict: Dict[str, int],
        num_layers: int,
        num_tasks_experts: int,
        num_shared_experts: int,
        expert_units: List[int],
        expert_activation: str = "leakyrelu:0.1",
        expert_dropout_rate: Optional[float] = None,
        expert_use_bn: bool = False,
        tower_units: List[int] = None,
        tower_activation: Optional[str] = "leakyrelu:0.1",
        tower_dropout_rate: Optional[float] = None,
        tower_use_bn: Optional[bool] = None,
    ):
        """
        Args:
            inputs_dim (int): The dimension of the input features.
            target_dict (Dict[str, int]): The dictionary of target names and their dimensions.
                e.g., {"AIRFLOW": 4, "PRESENCE": 6}
            num_layers (int): The number of ExtractionNet layers.
            num_tasks_experts (int): The number of task experts.
            num_shared_experts (int): The number of shared experts.
            expert_units (List[int]): The hidden units of experts. e.g., [5, 6, 7]
            expert_activation (str): The activation function of experts.
            expert_dropout_rate (Optional[float]): The dropout rate of experts.
            expert_use_bn (bool): Whether to use batch normalization in experts.
            tower_units (Optional[List[int]]): The hidden units of towers.
            tower_activation (Optional[str]): The activation function of towers.
            tower_dropout_rate (Optional[float]): The dropout rate of towers.
            tower_use_bn (Optional[bool]): Whether to use batch normalization in towers.
        """
        super().__init__()
        self.extractionNet_layers = torch.nn.ModuleList(
            [
                ExtractionNet(
                    inputs_dim if layer_id == 0 else expert_units[-1],
                    target_dict,
                    num_tasks_experts,
                    num_shared_experts,
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

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        for layer in self.extractionNet_layers:
            _outputs = layer(inputs)
            inputs = _outputs
        return _outputs  # [task1_out, task2_out, ...]
