"""
* 单目标、基于神经网络的遗传 problem 类
* 该 problem：SN1
*       * 决策变量：3 个风机频率的十倍 + 7 个送风阀开度 + 4 个回风阀开度 + 4 个排风阀开度
*       * 目标函数：房间压差的 RMSE 与风机频率均值的综合指标
*           0.5 * RMSE + 0.5 * (1 - 风机频率均值)
*       * 约束条件：总送风量、各房间压差
*
* File: MyProblem.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-15 22:25:50
* ----------------------------
* Modified: 2024-03-25 21:02:32
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
from pop_history import pop_history


class SN1(ea.Problem):  # 继承 Problem 父类
    def __init__(self, model: torch.nn.Module, scaler_path: tuple[str] = None):
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

        with open(scaler_path[0], "rb") as f:
            scaler_x = pickle.load(f)
        with open(scaler_path[1], "rb") as f:
            scaler_y = pickle.load(f)
        self.scalers = scaler_x, scaler_y

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
        ref_pres = np.array([10, 15, 32, 32, 30, 25])  # 房间压差设计值
        ref_total_SV = 3760  # 总送风量设计参考值
        ref_total_OV = 2312  # 新风量设计参考值
        ref_total_RV = 1448  # 回风量设计参考值
        ref_total_EV = 1600  # 总排风量设计参考值

        # ---------- Calculate objective function value ---------- #
        # features: decision variables + controlled variables
        # order as the end of this file
        features = self._model_eval(pop.Phen)  # (num_pop, 28)

        # 参照文件末尾的变量顺序
        mau_freq = features[:, 0].reshape(-1, 1)
        # ahu_freq = features[:, 1].reshape(-1, 1)
        # ef_freq = features[:, 2].reshape(-1, 1)
        freq_mean = np.mean(features[:, :3], axis=1).reshape(-1, 1)

        room_pres = features[:, -6:]
        room_pres_err = np.abs(room_pres - ref_pres)  # 逐房间压差偏差
        room_pres_rmse = np.sqrt(
            np.mean(np.square(room_pres - ref_pres), axis=1)
        ).reshape(-1, 1)

        pop.ObjV = np.hstack([freq_mean + room_pres_rmse / 10])

        # for multi-objective optimization
        # pop.ObjV = np.hstack([mau_freq, ahu_freq, ef_freq, room_pres_err])

        # ----------------- Calculate Constraint ----------------- #
        # 约束函数值为负表示满足约束，为正表示不满足约束，越大表示不满足约束程度越大
        total_RV = features[:, -7]
        total_EV = features[:, -8]
        total_SV = features[:, -9]
        total_OV = features[:, -10]

        # # 送风量约束，偏差不超过 10%
        # c_TSV = np.abs(total_SV - ref_total_SV) / ref_total_SV - 0.05

        # 送风量约束，大于设计值符合约束
        c_TSV = ref_total_SV - total_SV

        # 送风量约束，小于设计值 1.05 倍符合约束
        # c_TSV_2 = total_SV - ref_total_SV * 1.05

        # c_TOV = ref_total_OV - total_OV
        # c_TOV = np.abs(total_OV - ref_total_OV) / ref_total_OV - 0.05

        # c_TRV = ref_total_RV - total_RV
        # c_TRV = np.abs(total_RV - ref_total_RV) / ref_total_RV - 0.05

        c_TEV = ref_total_EV - total_EV
        # c_TEV = total_EV - ref_total_EV * 1.1
        # c_TEV = np.abs(total_EV - ref_total_EV) / ref_total_EV - 0.05

        # 房间压差约束，任意房间偏差不超过 5 Pa
        c_RP = np.max(room_pres_err, axis=1) - 3

        # mau 频率约束，不小于 10Hz
        # c_MAU = 10 - mau_freq

        pop.CV = np.column_stack([c_RP, c_TSV, c_TEV])
        # ----------------- Save history data ----------------- #
        pop_history["average_freq_mean"].append(np.mean(freq_mean))
        pop_history["best_freq_mean"].append(np.min(freq_mean))

    def _model_eval(self, Phen):
        """基于神经网络代理模型，使用决策变量计算受控变量。

        pop.Phen -> metrics:dict -> pop.ObjV
                                 -> pop.CV

        Args:
            Phen (ndarray): 决策变量矩阵 (NIND, Dim)，行：逐个体，列：逐决策变量
            scaler_path (list[str]): 两个 scaler 的路径，[x_scaler_path, y_scaler_path]

        Returns:
            features (ndarray): 所有特征的矩阵，行：逐个体，列：逐特征，（可控 + 受控）
        """
        scaler_x = self.scalers[0]
        scaler_y = self.scalers[1]

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
        _model_out = torch.concat(_model_out, dim=1)
        # Controlled variables
        _ctrl_var = np.vstack(
            scaler_y.inverse_transform(_model_out.detach()).cpu().numpy()
        )

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
    "TOT_FRSH_VOL",
    "TOT_SUPP_VOL",
    "TOT_EXH_VOL",
    "TOT_RET_VOL",
    "RM1_PRES",
    "RM2_PRES",
    "RM3_PRES",
    "RM4_PRES",
    "RM5_PRES",
    "RM6_PRES",
]
