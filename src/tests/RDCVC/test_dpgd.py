"""
* 测试 DPGD 指标
*
* File: test_dpgd.py
* Author: Fan Kai
* Soochow University
* Created: 2024-07-14 21:13:36
* ----------------------------
* Modified: 2024-07-14 23:38:29
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

# 使用拉丁超立方策略生成 1000 个列表，每个列表包含 6 个元素，每个元素在 [0, 100] 之间
import numpy as np
from pyDOE2 import lhs

from src.RDCVC.utils.dpgd import DPGD, CleanroomData

DESIGN_PRESSURE = [10, 15, 32, 32, 30, 25]  # 房间压差设计值
DESIGN_DATA = CleanroomData(
    rooms=["a", "b", "d", "e", "f", "c", "O"],
    room_pressures=DESIGN_PRESSURE + [0],
    room_relations=[
        ("d", "c"),
        ("e", "c"),
        ("e", "d"),
        ("f", "c"),
        ("c", "O"),
        ("c", "b"),
        ("b", "a"),  # b -> a
        ("a", "O"),
    ],
)


def test_dpgd_equal():
    room_pres = np.array(DESIGN_PRESSURE + [0])
    dpgd = _cal_DPGD(room_pres)
    assert dpgd == 0.0


def test_dpgd_bound(numb_samples: int = 1000):
    samples = generate_samples_LHS(numb_samples)
    dpgds = np.apply_along_axis(
        _cal_DPGD,
        axis=1,
        arr=np.hstack([samples, np.zeros((samples.shape[0], 1))]),  # 末尾添加 0
    ).reshape(-1, 1)

    print(f"样本数量：{numb_samples}，最大值：{dpgds.max()}，最小值：{dpgds.min()}")


def _cal_DPGD(room_pres: np.ndarray):
    "operates on 1-D arrays"
    data_compare = CleanroomData(
        rooms=["a", "b", "d", "e", "f", "c", "O"],
        room_pressures=room_pres,
        room_relations=[
            ("d", "c"),
            ("e", "c"),
            ("e", "d"),
            ("f", "c"),
            ("c", "O"),
            ("c", "b"),
            ("b", "a"),  # b -> a
            ("a", "O"),
        ],
    )
    return DPGD(
        DESIGN_DATA,
        data_compare,
        k_vertice=1,
        k_edge=2,
        timeout=1,
        show_err=False,
    )


def generate_samples_LHS(n_samples: int) -> np.ndarray:
    """按照 LHS 方法生成采样点

    Args:
        n_samples (int): 生成采样点的数量

    Returns:
        np.ndarray: 采样点
    """

    n_dim = 6  # 定义输入维度

    # 定义每个维度的范围
    bounds = [[0, 100]] * 6

    # 生成 LHS 采样样本
    samples = lhs(n_dim, samples=n_samples, criterion="corr")

    # 将样本范围映射到输入空间
    for i in range(n_dim):
        samples[:, i] = np.round(
            samples[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        ).astype(int)

    return samples


if __name__ == "__main__":
    test_dpgd_bound(100000)  # [0, 24]
