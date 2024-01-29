"""
* 单目标、基于神经网络的遗传 problem 类
* 该 problem：SN1
*       * 决策变量：3 个风机频率的十倍 + 7 个送风阀开度 + 4 个回风阀开度 + 4 个排风阀开度
*       * 目标函数：房间压差的 RMSE 与风机频率均值的综合指标
*           0.5 * RMSE + 0.5 * (1 - 风机频率均值)
*       * 约束条件：总送风量、各房间压差
*
* File: soea_nn_normal.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-15 22:25:50
* ----------------------------
* Modified: 2024-01-07 10:26:31
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
*
* 2024-01-06 20:07:17	FK  initialized main script (untested).
"""

import pickle

import geatpy as ea
import numpy as np
import torch


class SN1(ea.Problem):  # 继承 Problem 父类
    def __init__(self, model: torch.nn.Module):
        """初始化"""
        assert isinstance(
            model, torch.nn.Module
        ), "model must be a torch.nn.Module object."

        name = "SN1"  # 初始化 name（函数名称，可以随意设置），影响后续结果的保存

        # -------------------------- 目标 -------------------------- #
        M = 1  # 目标维数、f 的数量
        maxormins = [1] * M  # 目标最小最大化标记列表，1：min；-1：max

        # ------------------------- 决策变量 ------------------------- #
        Dim = 18  # 决策变量维数
        varTypes = [1 for _ in range(Dim)]  # 元素连续性（0:连续；1：离散）
        lb = [1 for _ in range(Dim)]  # 下边界值
        ub = [500, 500, 500] + [90 for _ in range(15)]  # 上边界值
        lbin = [0 for _ in range(Dim)]  # 下边界条件（0 不包含，1 表示包含）
        ubin = [0 for _ in range(Dim)]  # 上边界条件（0 不包含，1 表示包含）
        super().__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

        self.model = model.eval()  # 加载神经网络模型

    def aimFunc(self, pop):
        """
        目标函数

        目标函数计算方法：
            1. 计算房间压差
            2. 求房间压差与设计值的 RMSE
            3. 将 RMSE 值作为目标函数值

        Args:
            pop (Population): 一个种群对象，里面包含了种群的所有信息
        """
        eval_ref = self.calReferObjV()  # 获取目标函数参考值
        ref_total_SV = 3760  # 总送风量设计参考值
        ref_total_EV = 1600  # 总排风量设计参考值

        # ---------- Calculate objective function value ---------- #
        # features: decision variables + controlled variables
        # order as the end of this file
        features = self._model_eval(pop.Phen)  # (num_pop, 28)

        mau_freq = features[:, 0].reshape(-1, 1)
        ahu_freq = features[:, 1].reshape(-1, 1)
        ef_freq = features[:, 2].reshape(-1, 1)
        room_pres = features[:, -6:]
        room_pres_err = np.abs(room_pres - eval_ref)  # 逐房间压差偏差
        total_SV = features[:, -10]
        total_EV = features[:, -7]
        pop.ObjV = np.hstack(
            [mau_freq, ahu_freq, ef_freq, room_pres_err]
        )  # TODO: CORRECT HERE

        # ----------------- Calculate Constraint ----------------- #
        # 约束函数值为负表示满足约束，为正表示不满足约束，越大表示不满足约束程度越大
        # 送风量约束，偏差不超过 10%
        c_TSV = np.abs(total_SV - ref_total_SV) / ref_total_SV - 0.1
        c_TEV = np.abs(total_EV - ref_total_EV) / ref_total_EV - 0.1  # 排风
        pop.CV = np.column_stack([c_TSV, c_TEV])  # TODO: CORRECT HERE

    def _model_eval(self, Phen, scaler_path: list[str] = None):
        """基于神经网络代理模型，使用决策变量计算受控变量。

        pop.Phen -> metrics:dict -> pop.ObjV
                                 -> pop.CV

        Args:
            Phen (ndarray): 决策变量矩阵，行：逐个体，列：逐决策变量
            scaler_path (list[str]): 两个 scaler 的路径，[x_scaler_path, y_scaler_path]

        Returns:
            features (ndarray): 所有特征的矩阵，行：逐个体，列：逐特征，（可控 + 受控）
        """
        with open(scaler_path[0], "rb") as f:
            scaler_x = pickle.load(f)
        with open(scaler_path[1], "rb") as f:
            scaler_y = pickle.load(f)

        # ----------------------- 准备种群决策变量 ----------------------- #
        _dcsn_var = Phen.copy().astype(np.float64)  # Decision variables
        _dcsn_var[:, :3] /= 10.0  # 风机频率修正 (10 倍)
        # _num_pop = _dcsn_var.shape[0]  # 种群规模
        _model_in = scaler_x.transform(_dcsn_var)  # input matrix

        # ----------------------- 获取受控变量矩阵 ----------------------- #
        # for i in range(_num_pop):
        #     _x = _model_in[i, :]  # (18,)
        #     _y = self.model.predict(_x).reshape(-1)  # (10,)
        #     _ctrl_var.append(_y)
        _model_out = self.model(torch.Tensor(_model_in))
        # Controlled variables
        _ctrl_var = np.vstack(scaler_y.inverse_transform(_model_out.detach().numpy()))

        return np.hstack((_dcsn_var, _ctrl_var))


# 以下为决策（可控）变量和受控变量的顺序
ctrlable_vars = decision_vars = [
    "MAU_FREQ",
    "AHU_FREQ",
    "EF_FREQ",
    "RM1_SUPP_DMPR_0",
    "RM2_SUPP_DMPR_0",
    "RM3_SUPP_DMPR_0",
    "RM4_SUPP_DMPR_0",
    "RM5_SUPP_DMPR_0",
    "RM6_SUPP_DMPR_0",
    "RM6_SUPP_DMPR_1",
    "RM2_RET_DMPR_0",
    "RM3_RET_DMPR_0",
    "RM4_RET_DMPR_0",
    "RM6_RET_DMPR_0",
    "RM3_EXH_DMPR_0",
    "RM4_EXH_DMPR_0",
    "RM5_EXH_DMPR_0",
    "RM5_EXH_DMPR_1",
]
ctrled_vars = [
    "TOT_SUPP_VOL",
    "TOT_FRSH_VOL",
    "TOT_RET_VOL",
    "TOT_EXH_VOL",
    "RM1_PRES",
    "RM2_PRES",
    "RM3_PRES",
    "RM4_PRES",
    "RM5_PRES",
    "RM6_PRES",
]
