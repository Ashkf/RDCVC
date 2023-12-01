"""
*
*
* File: test_cvcnet.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-27 02:35:39
* ----------------------------
* Modified: 2023-12-01 02:21:19
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

from RDCVC.models.prediciton.CVCNet.models.cvcnet import (  # noqa: E402
    CVCNet,
    ExtractionNet,
)


@pytest.mark.parametrize("is_last_layer", [True, False, None])
@pytest.mark.parametrize("num_tasks_experts", [3, 5])
@pytest.mark.parametrize("num_shared_experts", [3, 5])
@pytest.mark.parametrize("expert_units", [[3] * 3, [5] * 3])
@pytest.mark.parametrize("tower_units", [[3] * 3, [5] * 3])
@pytest.mark.parametrize("use_bn", [True, False, None])
@pytest.mark.parametrize("dropout_rate", [0.0, 0.1, None])
# @pytest.mark.parametrize(
#     "is_last_layer,num_tasks_experts,num_shared_experts,use_bn,dropout_rate,expert_units,tower_units,",
#     [(True, 3, 4, False, None, [5, 6, 7], [8, 9, 10])],
# )
def test_ExtractionNet(
    is_last_layer,
    num_tasks_experts,
    num_shared_experts,
    use_bn,
    dropout_rate,
    expert_units,
    tower_units,
):
    inputs_dim = 18
    target_dict = {"PRESENCE": 6, "AIRFLOW": 4}
    eNet = ExtractionNet(
        inputs_dim=inputs_dim,
        target_dict=target_dict,
        num_tasks_experts=num_tasks_experts,
        num_shared_experts=num_shared_experts,
        expert_units=expert_units,
        expert_activation="leakyrelu:0.1",
        expert_dropout_rate=dropout_rate,
        expert_use_bn=use_bn,
        tower_units=tower_units,
        tower_activation="leakyrelu:0.1",
        tower_dropout_rate=dropout_rate,
        tower_use_bn=use_bn,
        is_last_layer=is_last_layer,
    )
    # Create dummy input tensor
    batch_size = 99
    inputs = torch.randn(batch_size, inputs_dim)

    # Call the forward method
    outputs = eNet.forward(inputs)

    assert eNet

    if is_last_layer:
        assert isinstance(outputs, dict)
        assert len(outputs) == len(target_dict)
        for k, v in outputs.items():
            assert isinstance(v, torch.Tensor)
            assert k in target_dict.keys()
            assert v.shape == (batch_size, target_dict[k])
    else:
        assert isinstance(outputs, list)
        assert len(outputs) == 4  # SA + RA + EA + Shared
        for v in outputs:
            assert isinstance(v, torch.Tensor)
            assert v.shape == (batch_size, expert_units[-1])


@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("num_tasks_experts", [4, 5])
@pytest.mark.parametrize("num_shared_experts", [8, 9])
@pytest.mark.parametrize("expert_units", [[2] * 3, [5] * 3])
@pytest.mark.parametrize("tower_units", [[5] * 3, [6] * 3])
@pytest.mark.parametrize("use_bn", [True, False, None])
@pytest.mark.parametrize("dropout_rate", [0.0, 0.1, None])
def test_CVCNet(
    num_layers,
    num_tasks_experts,
    num_shared_experts,
    expert_units,
    tower_units,
    use_bn,
    dropout_rate,
):
    inputs_dim = 18
    target_dict = {"PRESENCE": 6, "AIRFLOW": 4}
    cNet = CVCNet(
        inputs_dim=inputs_dim,
        target_dict=target_dict,
        num_layers=num_layers,
        num_tasks_experts=num_tasks_experts,
        num_shared_experts=num_shared_experts,
        expert_units=expert_units,
        expert_activation="leakyrelu:0.1",
        expert_dropout_rate=dropout_rate,
        expert_use_bn=use_bn,
        tower_units=tower_units,
        tower_activation="leakyrelu:0.1",
        tower_dropout_rate=dropout_rate,
        tower_use_bn=use_bn,
    )
    # Create dummy input tensor
    batch_size = 99
    inputs = torch.randn(batch_size, inputs_dim)

    # Call the forward method
    outputs = cNet.forward(inputs)

    assert cNet
    assert isinstance(outputs, dict)
    assert len(outputs) == len(target_dict)
    for k, v in outputs.items():
        assert isinstance(v, torch.Tensor)
        assert k in target_dict.keys()
        assert v.shape == (batch_size, target_dict[k])
