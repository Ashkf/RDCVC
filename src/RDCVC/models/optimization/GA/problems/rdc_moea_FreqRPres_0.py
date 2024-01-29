"""
*
*
* File: rdc_moea_FreqRPres_0.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-15 02:25:50
* ----------------------------
* Modified: 2024-01-07 12:00:58
* Modified By: Fan Kai
* =======================================================LC_TIME=en_DK.UTF-8=================
* HISTORY:
"""

import geatpy as ea
import numpy as np

"""
    该 problem 为 RDC 压差控制默认方案
        * 决策变量：3 个风机频率的十倍 + 7 个送风阀开度 + 4 个回风阀开度 + 4 个排风阀开度
        * 目标函数：
            1. 每个风机频率
            2. 每个房间压差
        * 约束条件：系统送风 + 系统排风
"""


class RdcMoeaFRp0(ea.Problem):
    def __init__(self, model):
        M = 9  # 目标维数、f 的数量
        Dim = 18  # 决策变量维数
        # name：MainName + [目标函数类型] + [约束类型]
        name = "".join(["RdcMoea", "FRp", "0"])
        maxormins = [1] * M  # 目标最小最大化标记列表，1：min；-1：max
        varTypes = [1 for _ in range(Dim)]  # 0:元素是连续的；1：是离散的 p
        lb = [1 for _ in range(Dim)]  # 决策变量下界
        ub = [500, 500, 500] + [90 for _ in range(15)]  # 决策变量上界
        lbin = [0 for _ in range(Dim)]  # 下边界（0 不包含，1 表示包含）
        ubin = [0 for _ in range(Dim)]  # 上边界（0 不包含，1 表示包含）

        super().__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

        self.model = model  # 加载神经网络模型

    def aimFunc(self, pop):
        """
        目标函数

        存在两个主要功能：
            1. 计算目标函数值
            2. 计算约束函数值

        Args:
            pop (Population): 一个种群对象，里面包含了种群的所有信息
        """
        eval_ref = np.array([10, 15, 30, 35, 30, 25])  # 房间压差设计值
        ref_total_SV = 3760  # 总送风量设计参考值
        ref_total_EV = 1600  # 总排风量设计参考值

        # ---------- Calculate objective function value ---------- #
        metrics = self._model_eval(pop.Phen)  # (num_pop, 28)
        mau_freq = metrics[:, 0].reshape(-1, 1)
        ahu_freq = metrics[:, 1].reshape(-1, 1)
        ef_freq = metrics[:, 2].reshape(-1, 1)
        room_pres = metrics[:, -6:]
        room_pres_err = np.abs(room_pres - eval_ref)  # 逐房间压差偏差
        total_SV = metrics[:, -10]
        total_EV = metrics[:, -7]
        pop.ObjV = np.hstack([mau_freq, ahu_freq, ef_freq, room_pres_err])

        # ----------------- Calculate Constraint ----------------- #
        # 约束函数值为负表示满足约束，为正表示不满足约束，越大表示不满足约束程度越大

        # 送风量约束，偏差不超过 10%
        c_TSV = np.abs(total_SV - ref_total_SV) / ref_total_SV - 0.1
        c_TEV = np.abs(total_EV - ref_total_EV) / ref_total_EV - 0.1  # 排风
        pop.CV = np.column_stack([c_TSV, c_TEV])

    def _model_eval(self, Phen):
        """从遗传使用的 model 计算决策变量对应的受控变量。

        pop.Phen -> metrics:dict -> pop.ObjV
                                 -> pop.CV

        Args:
            Phen (ndarray): 决策变量矩阵，行：逐个体，列：逐决策变量

        Returns:
            metrics (ndarray): 指标矩阵，行：逐个体，列：逐指标
        """
        # ----------------------- 准备种群决策变量 ----------------------- #
        _dcsn_var = Phen.copy().astype(np.float64)  # 决策变量
        _dcsn_var[:, :3] /= 10.0  # 风机频率修正
        _num_pop = _dcsn_var.shape[0]

        # ----------------------- 获取受控变量矩阵 ----------------------- #
        _ctrl_var = []
        for i in range(_num_pop):
            _x = _dcsn_var[i, :]  # (18,)
            _y = self.model.predict(_x).reshape(-1)  # (10,)
            _ctrl_var.append(_y)
        _ctrl_var = np.vstack(_ctrl_var)

        return np.hstack((_dcsn_var, _ctrl_var))


# 以下为决策变量和受控变量的顺序
decision_vars = [
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
