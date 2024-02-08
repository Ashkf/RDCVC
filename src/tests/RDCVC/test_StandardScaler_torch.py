"""
*
*
* File: test_StandardScaler_torch.py
* Author: Fan Kai
* Soochow University
* Created: 2024-02-07 23:03:43
* ----------------------------
* Modified: 2024-02-09 00:04:07
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import os
import sys

import pytest
import torch
from sklearn.preprocessing import StandardScaler

# cur_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.insert(0, cur_path + "/../..")

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from RDCVC.models.prediciton.CVCNet.utils.standard_scaler_torch import (  # noqa: E402
    StandardScalerTorch,
)

# 生成测试数据
X_test = torch.tensor(
    [
        [0.6, -0.4, -0.8],
        [-1.3, -1.9, -1.4],
        [-0.8, -0.1, 0.2],
    ],
    dtype=torch.float32,
)

X_test = torch.tensor(
    [[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]],
    dtype=torch.float32,
)
mean_test = torch.mean(X_test, dim=0)
var_test = torch.var(X_test, dim=0, correction=False)
scale_test = torch.std(X_test, dim=0, correction=False)


@pytest.mark.parametrize("X", [X_test])
def test_fit(X):
    scaler = StandardScalerTorch().fit(X)
    assert isinstance(scaler, StandardScalerTorch) is True
    assert scaler.mean_.equal(mean_test)
    assert scaler.var_.equal(var_test)
    assert scaler.scale_.equal(scale_test)


@pytest.mark.parametrize("X", [X_test])
def test_reset(X):
    scaler = StandardScalerTorch().fit(X)
    scaler._reset()
    assert not hasattr(scaler, "mean_")
    assert not hasattr(scaler, "var_")
    assert not hasattr(scaler, "scale_")
    assert not hasattr(scaler, "n_samples_seen_")


@pytest.mark.parametrize("X", [X_test])
def test_transform(X):
    scaler = StandardScalerTorch().fit(X)
    X_transformed = scaler.transform(X)
    assert torch.allclose(X_transformed, (X - scaler.mean_) / scaler.scale_)


@pytest.mark.parametrize("X", [X_test])
def test_inverse_transform(X):
    scaler = StandardScalerTorch().fit(X)
    X_transformed = scaler.transform(X)
    X_inverse = scaler.inverse_transform(X_transformed)
    assert torch.allclose(X_inverse, X)


@pytest.mark.parametrize("X", [X_test])
def test_fit_transform(X):
    scaler = StandardScalerTorch().fit(X)
    X_transformed = scaler.transform(X)

    X_fit_transformed = StandardScalerTorch().fit_transform(X)

    assert torch.allclose(X_transformed, X_fit_transformed)


# compare with sklearn
@pytest.mark.parametrize("X", [X_test])
def test_vs_StandardScaler(X):
    X_transformed_torch = StandardScalerTorch().fit_transform(X)
    X_transformed_sklearn = StandardScaler().fit_transform(X.numpy())
    assert torch.allclose(X_transformed_torch, torch.tensor(X_transformed_sklearn))
