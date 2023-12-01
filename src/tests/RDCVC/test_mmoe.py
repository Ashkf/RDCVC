"""
*
*
* File: test_mmoe.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-29 09:08:49
* ----------------------------
* Modified: 2023-11-30 06:57:34
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import os
import sys

import torch

# cur_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.insert(0, cur_path + "/../..")

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from RDCVC.models.prediciton.CVCNet.models.mmoe import MMoE  # noqa: E402


def test_mmoe_forward_First_notLast():
    num_experts = 3
    expert_units = [18, 12, 13]
    mmoe = MMoE(
        inputs_dim=18,
        target_dict={"PRESENCE": 6, "AIRFLOW": 4},
        num_experts=num_experts,
        expert_units=expert_units,
        expert_activation="leakyrelu:0.1",
        expert_dropout_rate=None,
        expert_use_bn=False,
        tower_units=[13, 30, 20],
        tower_activation="leakyrelu:0.1",
        tower_dropout_rate=None,
        tower_use_bn=None,
        is_last_layer=False,
    )

    # Create dummy input tensor
    batch_size = 99
    # inputs = torch.randn(batch_size, 18)
    inputs = [torch.randn(batch_size, 18)] * num_experts

    # Call the forward method
    outputs = mmoe.forward(inputs)

    # Check if the outputs dictionary is not empty
    assert outputs

    # Check if the outputs list has the correct number of elements
    assert len(outputs) == num_experts

    assert outputs[0].shape == torch.Size([batch_size, expert_units[-1]])


def test_mmoe_forward_FirstandLast():
    num_experts = 3
    target_dict = {"PRESENCE": 6, "AIRFLOW": 4}
    mmoe = MMoE(
        inputs_dim=18,
        target_dict=target_dict,
        num_experts=num_experts,
        expert_units=[18, 12, 13],
        expert_activation="leakyrelu:0.1",
        expert_dropout_rate=None,
        expert_use_bn=False,
        tower_units=[13, 30, 20],
        tower_activation="leakyrelu:0.1",
        tower_dropout_rate=None,
        tower_use_bn=None,
        is_last_layer=True,
    )

    # Create dummy input tensor
    batch_size = 99
    inputs = torch.randn(batch_size, 18)

    # Call the forward method
    outputs = mmoe.forward(inputs)

    # Check if the outputs dictionary is not empty
    assert outputs

    # Check if the outputs list has the correct number of elements
    assert len(outputs) == len(target_dict)

    assert outputs["PRESENCE"].shape == torch.Size([batch_size, 6])
    assert outputs["AIRFLOW"].shape == torch.Size([batch_size, 4])


def test_mmoe_forward_OnlyMid():
    num_experts = 3
    expert_units = [18, 12, 13]
    target_dict = {"PRESENCE": 6, "AIRFLOW": 4}
    mmoe = MMoE(
        inputs_dim=18,
        target_dict=target_dict,
        num_experts=num_experts,
        expert_units=expert_units,
        expert_activation="leakyrelu:0.1",
        expert_dropout_rate=None,
        expert_use_bn=False,
        tower_units=[20, 30, 20],
        tower_activation="leakyrelu:0.1",
        tower_dropout_rate=None,
        tower_use_bn=None,
        is_last_layer=False,
    )

    # Create dummy input tensor
    batch_size = 99
    inputs = [torch.randn(batch_size, 18)] * num_experts

    # Call the forward method
    outputs = mmoe.forward(inputs)

    # Check if the outputs dictionary is not empty
    assert outputs

    # Check if the outputs list has the correct number of elements
    assert len(outputs) == num_experts

    assert outputs[0].shape == torch.Size([batch_size, expert_units[-1]])


def test_mmoe_forward_asLast():
    num_experts = 3
    mmoe = MMoE(
        inputs_dim=18,
        target_dict={"PRESENCE": 6, "AIRFLOW": 4},
        num_experts=num_experts,
        expert_units=[18, 12, 13],
        expert_activation="leakyrelu:0.1",
        expert_dropout_rate=None,
        expert_use_bn=False,
        tower_units=[13, 30, 20],
        tower_activation="leakyrelu:0.1",
        tower_dropout_rate=None,
        tower_use_bn=None,
        is_last_layer=True,
    )

    # Create dummy input tensor
    batch_size = 99
    inputs = torch.randn(batch_size, 18)

    # Call the forward method
    outputs = mmoe.forward(inputs)

    # Check if the outputs dictionary is not empty
    assert outputs

    # Check if the outputs list has the correct number of elements
    assert len(outputs) == 2

    assert outputs["PRESENCE"].shape == torch.Size([batch_size, 6])
    assert outputs["AIRFLOW"].shape == torch.Size([batch_size, 4])
