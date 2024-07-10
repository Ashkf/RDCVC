"""
* 
* 
* File: CRITIC.py
* Author: Fan Kai
* Soochow University
* Created: 2024-07-04 23:00:35
* ----------------------------
* Modified: 2024-07-04 23:10:27
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""


# 使用客观赋权法 CRITIC 计算权衡解
import numpy as np
from skcriteria import mkdm
from skcriteria.preprocessing.weighters import CRITIC

# decision_matrix = NDSet.ObjV  # 2D array-like, shape (n_samples, n_criteria)
decision_matrix = np.array(
    [
        [128, 21, 3311],
        [234, 32, 2363],
        [563, 12, 999],
        [235, 74, 1223],
    ],
    dtype=float,
)


def dimensionless(x, objectives):
    # 无量纲化处理（正/逆向化）
    # 若该指标越大越好（正向指标），则 x' = (x - x_min) / (x_max - x_min)
    # 若该指标越小越好（负向指标），则 x' = (x_max - x) / (x_max - x_min)

    # x 应当为 2D array-like, shape (n_samples, n_criteria)
    if x.shape[1] != len(objectives):
        raise ValueError("dimensionless. dimension mismatch.")
    for i, obj in enumerate(objectives):
        if obj == max:
            _min_e = np.min(x[:, i])
            _max_e = np.max(x[:, i])
            _range = _max_e - _min_e
            A = x[:, i] - _min_e
            x[:, i] = (x[:, i] - _min_e) / _range
        elif obj == min:
            _min_e = np.min(x[:, i])
            _max_e = np.max(x[:, i])
            _range = _max_e - _min_e
            A = _max_e - x[:, i]
            x[:, i] = (_max_e - x[:, i]) / _range
        else:
            raise ValueError("dimensionless. invalid objective.")

    return x


dd_matrix = dimensionless(decision_matrix, [min, max, min])
print(f"dimensionless Decision matrix:\n{dd_matrix}")

dm = mkdm(
    dd_matrix,
    [max, max, max],
    criteria=["E", "DPerr", "ACHerr"],
)

# # 手动计算 CRITIC 权重
# def m_CRITIC(dm):
#     # 计算 CRITIC 权重
#     # 1. 计算 CRITIC 矩阵
#     _critic_matrix = np.zeros_like(dm)
#     for i in range(dm.shape[0]):
#         for j in range(dm.shape[1]):
#             _critic_matrix[i, j] = np.sum(dm[:, j] / dm[i, j])
#     # 2. 计算 CRITIC 权重
#     _weights = np.sum(_critic_matrix, axis=0)
#     _weights /= np.sum(_weights)
#     return _weights

dm = CRITIC().transform(dm)
print(f"Decision matrix:\n{dm},weights:\n{dm.weights}")
