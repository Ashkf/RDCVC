"""
* 基于图的压差梯度差异性指标（DPGD）计算
*
* File: dpgd.py
* Author: Fan Kai
* Soochow University
* Created: 2024-07-13 17:57:14
* ----------------------------
* Modified: 2024-07-14 15:14:25
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
*
* 2024-07-14 15:14:16	FK	实现
"""


from functools import partial

import numpy as np
from networkx import DiGraph, graph_edit_distance
from pydantic import BaseModel, field_validator, model_validator
from typing_extensions import Self


class CleanroomData(BaseModel):
    """多区域洁净室压差与渗透方向数据

    Raises:
        ValueError: rooms 和 room_pressures 应该等长
        ValueError: room_relations 中的元素必须是 rooms 中的元素
        ValueError: room_relations 中不应该出现双向关系

    Returns:
        CleanroomData: 多区域洁净室压差与渗透方向数据
    """

    rooms: list[str]  # [room_name, ...]
    room_pressures: list[float]  # [pressure, ...]
    room_relations: list[tuple[str, str]]  # [(rom1, rom2), ...], rom1 -> rom2

    @model_validator(mode="after")
    def check_rooms_len(self) -> Self:
        if len(self.rooms) != len(self.room_pressures):
            raise ValueError("rooms 和 room_pressures 应该等长")
        return self

    @model_validator(mode="after")
    def check_room_relations(self) -> Self:
        if not all(set(relation) <= set(self.rooms) for relation in self.room_relations):
            raise ValueError("room_relations 中的元素必须是 rooms 中的元素")
        return self

    @field_validator("room_relations")
    @classmethod
    def check_room_relations_reverse(cls, v):
        for relation in v:
            if relation[::-1] in v:
                raise ValueError("room_relations 中不应该出现双向关系")
        return v


def cal_ew_press(node1, node2):
    r"""计算两个节点之间的等效权重，若两个节点的压力差为负，则返回负值，否则返回正值。
    $$w_{ij} = \Delta p^{0.5}_{ij} = |p_{i} - p_{j}|^{0.5}$$
    """
    press_err = node1["pressure"] - node2["pressure"]

    if press_err < 0:
        rlt = -((-press_err) ** 0.5)
    else:
        rlt = press_err**0.5

    return rlt


def create_cleanroom_graph(
    data: CleanroomData,
    cal_edge_weight: callable,
    show_err=True,
) -> DiGraph:
    """创建洁净室压差与渗透气流网络图

    该函数接受一个 CleanroomData 实例，返回一个有向图。
    图中的节点代表洁净室，边代表洁净室之间的压差关系。
    需要传入一个计算边权重的函数，该函数接受两个节点，返回边权重。
    边权重固定为正值，若计算出的边权重为负值，则将边的方向反转。

    Args:
        data (CleanroomData): 包含洁净室压差与渗透方向数据的实例
        cal_edge_weight (callable): 计算边权重的函数

    Returns:
        nx.DiGraph: 洁净室压差与渗透气流网络图
    """
    G = DiGraph()
    G.add_nodes_from(
        [
            (room, {"pressure": pressure})
            for room, pressure in zip(data.rooms, data.room_pressures, strict=True)
        ]
    )
    for relation in data.room_relations:
        n1, n2 = relation
        weight = cal_edge_weight(G.nodes[n1], G.nodes[n2])

        if weight < 0:
            n1, n2 = n2, n1
            weight = -weight

            # 同时原地修改 data 中的 room_relations
            data.room_relations[data.room_relations.index(relation)] = (n1, n2)
            if show_err:
                print(f"Warning: {relation} 的压差权重为负，已反转为 {n1} -> {n2}")

        G.add_edge(n1, n2, weight=weight, nodes=(n1, n2))

    G.mean_pressure = np.mean(data.room_pressures)
    G.mean_edge_weight = np.mean([edge["weight"] for _, _, edge in G.edges(data=True)])
    return G


def count_reversed_edges(G1, G2) -> int:
    """计算两个工况图中逆压差梯度的边数。"""
    edges_G1 = set(G1.edges())
    edges_G2 = set(G2.edges())
    reversed_count = 0
    for u, v in edges_G1:
        if (v, u) in edges_G2:
            reversed_count += 1

    return reversed_count


def _node_subst_cost(node_d, node_c, G_design, k_vertice):
    r"""返回节点替换的开销。
        以节点属性字典为输入，返回一个正数作为替换开销。

        $$C_{\mathrm{vertice}}\left(p_{i}^{c}, p_{i}^{d}\right) \\
        =k_{\mathrm{vertice}} \cdot\left(\frac{p_{i}^{d}\cdot \\
        \left|p_{i}^{c}-p_{i}^{d}\right|}{\overline{p^d}^2}\right)^{T}$$

        node_subst_cost(G1.nodes[n1], G2.nodes[n2]) -> float
        """
    p_d = node_d["pressure"]
    p_c = node_c["pressure"]
    p_d_mean = G_design.mean_pressure
    cost = k_vertice * (p_d * abs(p_c - p_d)) / p_d_mean**2
    return cost


def _edge_subst_cost(edge_d, edge_c, G_design, k_edge):
    r"""返回边替换的开销。忽略边的方向。
        以边属性字典为输入，返回一个正数作为替换开销。

        $$C_{\mathrm{edge} }\left(w_{i j}^{c}, w_{i j}^{d}\right) \\
        =k_{\mathrm{edge} } \cdot\left(\frac{\max{(p_i^{d},p_j^{d})} \\
        \cdot \left|w_{i j}^{c}-w_{i j}^{d}\right|}{\overline{w^{d}}^3}\right)^{T}$$

        edge_subst_cost(G1.edges[e1], G2.edges[e2]) -> float
        """

    w_d = edge_d["weight"]
    w_c = edge_c["weight"]
    w_d_mean = G_design.mean_edge_weight
    n1, n2 = edge_d["nodes"]
    p_i_d, p_j_d = (
        G_design.nodes[n1]["pressure"],
        G_design.nodes[n2]["pressure"],
    )
    cost = k_edge * (max(p_i_d, p_j_d) * abs(w_c - w_d)) / w_d_mean**3
    return cost


def DPGD_by_graph(
    G_design: DiGraph,
    G_compare: DiGraph,
    node_subst_cost: callable,
    edge_subst_cost: callable,
    timeout=5,
) -> float:
    """计算压差梯度差异性指标。

    本质是对 nx graph_edit_distance 函数的封装。

    逆压差梯度：
    当逆压差梯度发生时，对应节点间边的方向相反（权重固定为正实数），需要特别处理。
    此时应当有 $I_{\rm{DPGD}} = -1$。每再有一条逆压差梯度的边，$I_{\rm{DPGD}} - 1$。
    """
    reversed_count = count_reversed_edges(G_design, G_compare)
    if reversed_count > 0:
        return -reversed_count

    return graph_edit_distance(
        G_design,
        G_compare,
        node_subst_cost=node_subst_cost,
        edge_subst_cost=edge_subst_cost,
        timeout=timeout,
    )


def DPGD(
    data_design: CleanroomData,
    data_compare: CleanroomData,
    k_vertice=1,
    k_edge=2,
    cal_edge_weight=cal_ew_press,
    timeout=2,
    show_err=True,
) -> float:
    G_design = create_cleanroom_graph(data_design, cal_edge_weight, show_err=show_err)
    G_compare = create_cleanroom_graph(data_compare, cal_edge_weight, show_err=show_err)

    node_subst_cost = partial(
        _node_subst_cost,
        G_design=G_design,
        k_vertice=k_vertice,
    )

    edge_subst_cost = partial(
        _edge_subst_cost,
        G_design=G_design,
        k_edge=k_edge,
    )

    return DPGD_by_graph(
        G_design,
        G_compare,
        node_subst_cost=node_subst_cost,
        edge_subst_cost=edge_subst_cost,
        timeout=timeout,
    )


if __name__ == "__main__":
    data_d = CleanroomData(
        rooms=["a", "b", "c"],
        room_pressures=[100, 110, 121],
        room_relations=[("b", "a"), ("c", "b")],
    )

    data_c = CleanroomData(
        rooms=["a", "b", "c"],
        room_pressures=[100, 120, 121],
        room_relations=[("b", "a"), ("c", "b")],
    )

    print(DPGD(data_d, data_c))
