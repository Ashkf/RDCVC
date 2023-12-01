"""
*
*
* File: test_multi-layer_mmoe.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-30 07:01:11
* ----------------------------
* Modified: 2023-11-30 08:23:11
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import os
import sys

import pytest
import torch

# cur_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.insert(0, cur_path + "/../..")

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from RDCVC.models.prediciton.CVCNet.models.mmoe import ML_MMoE  # noqa: E402


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_mmmoe_forward(num_layers):
    inputs_dim = 18
    target_dict = {"PRESENCE": 6, "AIRFLOW": 4}
    num_layers = num_layers
    num_experts = 3
    expert_units = [18, 12, 13]
    expert_activation = "leakyrelu:0.1"
    expert_dropout_rate = None
    expert_use_bn = False
    tower_units = [13, 30, 20]
    tower_activation = "leakyrelu:0.1"
    tower_dropout_rate = None
    tower_use_bn = False
    mmmoe = ML_MMoE(
        inputs_dim=inputs_dim,
        target_dict=target_dict,
        num_layers=num_layers,
        num_experts=num_experts,
        expert_units=expert_units,
        expert_activation=expert_activation,
        expert_dropout_rate=expert_dropout_rate,
        expert_use_bn=expert_use_bn,
        tower_units=tower_units,
        tower_activation=tower_activation,
        tower_dropout_rate=tower_dropout_rate,
        tower_use_bn=tower_use_bn,
    )

    # Create dummy input tensor
    batch_size = 99
    # inputs = torch.randn(batch_size, 18)
    inputs = torch.randn(batch_size, 18)

    # Call the forward method
    outputs = mmmoe.forward(inputs)

    # Check if the outputs dictionary is not empty
    assert outputs
    assert isinstance(outputs, dict)

    # Check if the outputs list has the correct number of elements
    assert len(outputs) == len(target_dict)

    assert outputs["PRESENCE"].shape == torch.Size([batch_size, 6])
    assert outputs["AIRFLOW"].shape == torch.Size([batch_size, 4])
