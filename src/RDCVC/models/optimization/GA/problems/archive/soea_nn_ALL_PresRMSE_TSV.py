# -*- coding: utf-8 -*-
import os
import pickle

import geatpy as ea
import numpy as np
import torch
from CoreGA.utils.DataProcessor import DataProcessor
from sklearn.metrics import mean_squared_error as MSE

"""
    该 problem：
        * 决策变量：3 个风机频率的十倍 + 7 个送风阀开度 + 4 个回风阀开度 + 4 个排风阀开度
        * 目标函数：bim 模型的结果（房间压差）与设计（房间压差）的 RMSE
        * 约束条件：总送风量
"""


class MyProblem(ea.Problem):  # 继承 Problem 父类
    def __init__(self, nndir="default"):
        """初始化"""
        # name 的一般命名规则：MainName + [决策变量类型] + [目标函数类型] + [约束类型]
        name = "GAProblem-NN[ALL][PresRMSE][TSV]"  # 初始化 name（函数名称，可以随意设置），影响后续结果的保存
        M = 1  # 初始化 M（目标维数、f 的数量）
        maxormins = [1]  # 初始化 maxormins（目标最小最大化标记列表，1：min；-1：max）
        Dim = 18  # 初始化 Dim（决策变量维数）
        # =========================== 以下为决策变量的设置 ===========================
        varTypes = [
            1 for _ in range(Dim)
        ]  # 元素为 0 表示对应的变量是连续的；1 表示是离散的
        lb = [1 for _ in range(Dim)]  # 决策变量下界
        ub = [500, 500, 500] + [90 for _ in range(15)]  # 决策变量上界
        lbin = [
            0 for _ in range(Dim)
        ]  # 决策变量下边界（0 表示不包含该变量的下边界，1 表示包含）
        ubin = [
            0 for _ in range(Dim)
        ]  # 决策变量上边界（0 表示不包含该变量的上边界，1 表示包含）
        # =========================== 以上为决策变量的设置 ==========================
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        print(f"{name} initialized.")
        self.nndir = nndir  # 神经网络模型所在的文件夹
        self.nnmodel = self.prepare_nnmodel()  # 加载神经网络模型

    def prepare_nnmodel(self):
        """从文件夹中加载神经网络模型"""
        root_dir = self.nndir
        model_path = os.path.join(root_dir, "final_model.pth")
        # 尝试从文件夹中加载神经网络模型
        if not os.path.exists(model_path):
            print("No model found in the folder!")
            exit(1)

        model = torch.load(model_path)  # 加载神经网络模型
        return model

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
        Vars = (
            pop.Phen
        )  # 得到决策变量矩阵，这里的 Vars 是一个二维矩阵，每一行对应一个个体，每一列对应一个决策变量
        # ===================== 以下是初始化约束函数值 =====================
        # 由于约束条件从计算结果中提取，所以初始化约束函数值矩阵为全 0 矩阵，（种群规模行，1 列）
        c2 = np.zeros((Vars.shape[0], 1))  # 第 2 个约束
        c3 = np.zeros((Vars.shape[0], 1))  # 第 3 个约束
        c4 = np.zeros((Vars.shape[0], 1))  # 第 4 个约束
        c5 = np.zeros((Vars.shape[0], 1))  # 第 5 个约束
        c6 = np.zeros((Vars.shape[0], 1))  # 第 6 个约束
        c7 = np.zeros((Vars.shape[0], 1))  # 第 7 个约束
        # ===================== 以下是计算目标函数值 =====================
        eval_ref = self.calReferObjV()  # 获取目标函数参考值
        total_SV = np.zeros((Vars.shape[0], 1))  # 总送风量初始化为全 0 矩阵
        pop.ObjV = np.zeros((Vars.shape[0], 1))  # 种群的目标函数值矩阵初始化为全 0 矩阵
        for i in range(Vars.shape[0]):
            """该循环依照行对种群中的每个个体进行处理，得到每个个体的目标函数值"""
            # X 是一个个体字典，键是决策变量的名称，值是决策变量的值
            X = DataProcessor.Vars2Dict(
                Vars[i, :]
            )  # 将决策变量矩阵中的第 i 行转换为一个个体的字典
            result = self.get_result(X)  # 获取决策变量输入对应的输出

            # ===================== 从 result 中取出指标所需的变量  =====================
            #  从 result 中取出用于计算目标函数的变量的值
            room_pres = result[:, :-1].reshape(7, -1)  # 房间压差
            total_SV[i, 0] = result[:, -1]  # 总送风量
            # 计算目标函数值 (RMSE)，赋值给 pop 种群对象的 ObjV 属性
            pop.ObjV[i, 0] = np.sqrt(MSE(room_pres, eval_ref))
            # 按照房间的压差与设计值比较，计算约束函数值
            c2[i, 0] = np.abs(room_pres[0] - 10) - 10  # RM1_PRES 偏差不超过 1 Pa
            c3[i, 0] = np.abs(room_pres[1] - 15) - 10  # RM2_PRES 偏差不超过 1 Pa
            c4[i, 0] = np.abs(room_pres[2] - 30) - 10  # RM3_PRES 偏差不超过 1 Pa
            c5[i, 0] = np.abs(room_pres[3] - 35) - 10  # RM4_PRES 偏差不超过 1 Pa
            c6[i, 0] = np.abs(room_pres[4] - 30) - 10  # RM5_PRES 偏差不超过 1 Pa
            c7[i, 0] = np.abs(room_pres[5] - 25) - 10  # RM6_PRES 偏差不超过 1 Pa

        # ===================== 以下是计算约束 =====================
        # 采用可行性法则处理约束，numpy 的 hstack() 把列向量拼成 CV 矩阵
        # 计算约束函数值，赋值给 pop 种群对象的 CV 属性
        # 约束函数值为负表示满足约束，为正表示不满足约束，越大表示不满足约束程度越大
        # 把 c1、c2、...等列向量拼成 CV 矩阵，赋值给种群
        # -------------------------------------------------------
        ref_total_supply = 3760  # 总风量设计参考值
        c1 = (
            np.abs(total_SV - ref_total_supply) / ref_total_supply - 0.1
        )  # 送风量约束，偏差不能超过 10%
        pop.CV = np.hstack([c1, c2, c3, c4, c5, c6, c7])

    def get_result(self, X) -> dict:
        """获取输入变量 X 对应的输出

        Args:
            X (dict): 个体的决策变量

        Returns:
            result
        """
        self.nnmodel.eval()
        item = DataProcessor.X2Tensor(X)
        item.requires_grad_(False)
        # ============= 归一化 =============
        scaler_path = os.path.join(self.nndir, "scalers_dict.pkl")  # 归一化器路径
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
        items_scalered = scalers["scaler_in"].transform(item)
        # ============= 预测 =============
        pred = self.nnmodel(torch.Tensor(items_scalered))
        # ============= 反归一化 =============
        pred = scalers["scaler_out"].inverse_transform(pred.detach().numpy())
        return pred

    def calReferObjV(self):
        """设定目标数参考值（理论最优值）

        用于计算或读取目标函数参考值，常常用待优化问题的理论全局最优解作为该参考值，用于后续进行评价进化结果的好坏。
        如果并不知道理论全局最优解是什么，可以不设置 calReferObjV() 函数。

        用于计算参考点（reference point）的适应度值，主要应用于多目标优化问题（MOEA）中的 NSGA-II 算法。
        NSGA-II 算法使用参考点来度量种群的逼近度，从而保证种群的分布均匀且具有多样性。

        Returns:
            np.array: 目标函数参考值组成的一维数组
        """
        ref_value = {
            "一更": 10,
            "二更": 15,
            "测试间一": 30,
            "测试间二": 35,
            "测试间三": 30,
            "洁净走廊": 25,
            "外走道": 0.0,
        }
        order = [
            "一更",
            "二更",
            "测试间一",
            "测试间二",
            "测试间三",
            "洁净走廊",
            "外走道",
        ]  # 排序顺序
        ref_value = DataProcessor.dicts_to_ndarray(ref_value, order=order)  # 参考风量

        return ref_value.reshape(-1, 1)
