# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import mean_squared_error as MSE
import geatpy as ea
from CoreGA.utils.DampersProcessor import DampersProcessor  # bim 模型风阀操控接口
from CoreGA.utils.DataProcessor import DataProcessor
from utils.SQLiteManager import SQLiteManager  # 数据库接口

"""
    该 problem：
        * 决策变量：3 个风机频率的十倍 + 7 个送风阀开度 + 4 个回风阀开度 + 4 个排风阀开度
        * 目标函数：bim 模型的结果（房间压差）与设计（房间压差）的 RMSE
        * 约束条件：总送风量
"""


class MyProblem(ea.Problem):  # 继承 Problem 父类
    def __init__(self, nndir='default', store_data=False, use_existing_data=False):
        """初始化

        Args:
            store_data (bool, optional): 是否存储数据至数据库。Defaults to False.
            use_existing_data (bool, optional): 是否使用数据库已有数据。Defaults to False.
        """
        # name 的一般命名规则：MainName + [决策变量类型] + [目标函数类型] + [约束类型]
        name = 'GAProblem-soea-API[ALL][Pres][TEV-TSV]'  # 初始化 name（函数名称，可以随意设置），影响后续结果的保存
        M = 1  # 初始化 M（目标维数、f 的数量）
        maxormins = [1]  # 初始化 maxormins（目标最小最大化标记列表，1：min；-1：max）
        Dim = 18  # 初始化 Dim（决策变量维数）
        # =========================== 以下为决策变量的设置 ===========================
        varTypes = [1 for _ in range(Dim)]
        lb = [1 for _ in range(Dim)]
        ub = [500, 500, 500] + [90 for _ in range(15)]
        lbin = [0 for _ in range(Dim)]
        ubin = [0 for _ in range(Dim)]
        # =========================== 以上为决策变量的设置 ==========================
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

        self.damperProcessor = DampersProcessor()  # 实例化 bim 模型风阀操控接口
        self.DBM = SQLiteManager("data/shuimuBIM.sqlite3")  # 实例化数据库接口
        self.store_table_name = "SMKNRDC_bim_data"  # 存储数据的表名
        self.DBM.create_table(self.store_table_name)  # 创建表
        self.is_storing = store_data  # 是否存储数据
        self.is_using_existing_data = use_existing_data  # 是否使用已有数据
        if self.is_using_existing_data:
            self.DBM.load_table_cache(self.store_table_name)  # 加载整表数据作为缓存
        print(f"{name} initialized.")

    def aimFunc(self, pop):
        """
        目标函数

        目标函数计算方法：
            1. 计算 bim 模型的结果（实际房间压差）（这里的实际指的是 bim 模拟计算的值）
            2. 求实际房间压差与设计房间压差的 RMSE 值
            3. 将 RMSE 值作为目标函数值

        Args:
            pop (Population): 一个种群对象，里面包含了种群的所有信息
        """
        Vars = pop.Phen  # 得到决策变量矩阵，这里的 Vars 是一个二维矩阵，每一行对应一个个体，每一列对应一个决策变量
        # ===================== 以下是初始化约束函数值 =====================
        # # 由于约束条件从计算结果中提取，所以初始化约束函数值矩阵为全 0 矩阵，（种群规模行，1 列）
        c3 = np.zeros((Vars.shape[0], 1))  # 第 3 个约束；
        c4 = np.zeros((Vars.shape[0], 1))  # 第 4 个约束；
        c5 = np.zeros((Vars.shape[0], 1))  # 第 5 个约束；
        c6 = np.zeros((Vars.shape[0], 1))  # 第 6 个约束；
        c7 = np.zeros((Vars.shape[0], 1))  # 第 7 个约束；
        c8 = np.zeros((Vars.shape[0], 1))  # 第 8 个约束；
        # ===================== 以下是计算目标函数值 =====================
        ref_room_pres = np.array([10, 15, 30, 35, 30, 25, 0])  # 设计室内压力
        ref_room_supply_volume = np.array([235, 331, 689, 815, 722, 968])  # 设计房间送风量
        total_SUPP_VOL = np.zeros((Vars.shape[0], 1))  # 总送风量
        total_EXH_VOL = np.zeros((Vars.shape[0], 1))  # 总排风量
        pop.ObjV = np.zeros((Vars.shape[0], 1))  # 初始化种群的目标函数值矩阵
        for i in range(Vars.shape[0]):
            """该循环依照行对种群中的每个个体进行处理，得到每个个体的目标函数值"""
            # X 是一个个体字典，键是决策变量的名称，值是决策变量的值
            X = {"MAU_FREQ": Vars[i, 0] / 10,
                 "AHU_FREQ": Vars[i, 1] / 10,
                 "EF_FREQ": Vars[i, 2] / 10,
                 "RM1_SUPP_DMPR_0": Vars[i, 3],
                 "RM2_SUPP_DMPR_0": Vars[i, 4],
                 "RM3_SUPP_DMPR_0": Vars[i, 5],
                 "RM4_SUPP_DMPR_0": Vars[i, 6],
                 "RM5_SUPP_DMPR_0": Vars[i, 7],
                 "RM6_SUPP_DMPR_0": Vars[i, 8],
                 "RM6_SUPP_DMPR_1": Vars[i, 9],
                 "RM2_RET_DMPR_0": Vars[i, 10],
                 "RM3_RET_DMPR_0": Vars[i, 11],
                 "RM4_RET_DMPR_0": Vars[i, 12],
                 "RM6_RET_DMPR_0": Vars[i, 13],
                 "RM3_EXH_DMPR_0": Vars[i, 14],
                 "RM4_EXH_DMPR_0": Vars[i, 15],
                 "RM5_EXH_DMPR_0": Vars[i, 16],
                 "RM5_EXH_DMPR_1": Vars[i, 17]
                 }
            result = self.get_result(X)  # 个体 X 在 BIM 中的计算结果

            # ===================== 从 result 中取出指标所需的变量  =====================
            #  从 result 中取出取出房间压差
            room_pres_keys = ["RM1_PRES", "RM2_PRES", "RM3_PRES", "RM4_PRES", "RM5_PRES",
                              "RM6_PRES", "RM7_PRES"]
            room_pres = np.array([result.get(key) for key in room_pres_keys])
            #  从 result 中取出取出房间送风量
            room_supply_volume_keys = ["RM1_SUPP_VOL", "RM2_SUPP_VOL", "RM3_SUPP_VOL",
                                       "RM4_SUPP_VOL", "RM5_SUPP_VOL", "RM6_SUPP_VOL", ]
            room_supply_volume = np.array([result.get(key) for key in room_supply_volume_keys])
            # f1 = X['AHU_FREQ']  # 目标函数值：AHU 频率
            # f2 = X['MAU_FREQ']  # 目标函数值：MAU 频率
            # f3 = X['EF_FREQ']  # 目标函数值：EF 频率

            total_SUPP_VOL[i, 0] = result.get("TOT_SUPP_VOL")  # 计算种群总送风量
            total_EXH_VOL[i, 0] = result.get("TOT_EXH_VOL")  # 计算种群总排风量
            # 如果 room_pres 中有 nan 值，说明计算结果有问题，目标函数值设置为无穷
            if np.any(np.isnan(room_pres)) or np.any(np.isnan(total_SUPP_VOL)):
                pop.ObjV[i, 0] = np.inf  # np.inf 表示正无穷，-np.inf 表示负无穷，具体结合 problem 定义选取
                print(f'第 {(i + 1)} 个个体存在错误，目标函数值设置为 np.inf')
                continue
            # 计算目标函数值 (RMSE)，赋值给 pop 种群对象的 ObjV 属性
            pop.ObjV[i, 0] = np.sqrt(MSE(room_pres, ref_room_pres))
            # 按照房间的送风量与设计值比较，计算约束函数值
            c3[i, 0] = np.abs(room_supply_volume[0] - ref_room_supply_volume[0]) / room_supply_volume[0] - 0.2
            c4[i, 0] = np.abs(room_supply_volume[1] - ref_room_supply_volume[1]) / room_supply_volume[1] - 0.2
            c5[i, 0] = np.abs(room_supply_volume[2] - ref_room_supply_volume[2]) / room_supply_volume[2] - 0.2
            c6[i, 0] = np.abs(room_supply_volume[3] - ref_room_supply_volume[3]) / room_supply_volume[3] - 0.2
            c7[i, 0] = np.abs(room_supply_volume[4] - ref_room_supply_volume[4]) / room_supply_volume[4] - 0.2
            c8[i, 0] = np.abs(room_supply_volume[5] - ref_room_supply_volume[5]) / room_supply_volume[5] - 0.2

        # ===================== 以下是计算约束 =====================
        # 采用可行性法则处理约束，numpy 的 hstack() 把列向量拼成 CV 矩阵
        # 计算约束函数值，赋值给 pop 种群对象的 CV 属性
        # 约束函数值为负表示满足约束，为正表示不满足约束，越大表示不满足约束程度越大
        # 把 c1、c2、...等列向量拼成 CV 矩阵，赋值给种群
        # -------------------------------------------------------
        ref_total_supply = 3760  # 总风量设计参考值
        ref_total_exhaust = 1600  # 总排风量设计参考值
        c1 = ref_total_supply - total_SUPP_VOL  # 送风量约束，大于设计值
        c2 = ref_total_exhaust - total_EXH_VOL  # 排风量约束，大于设计值
        pop.CV = np.hstack([c1, c2, c3, c4, c5, c6, c7, c8])

    def get_result(self, X) -> dict:
        """针对某个个体获取 BIM 模拟结果

            1. 针对某个个体进行模拟
            2. 根据参数 is_using_existing_data 是否使用已经存在的数据进行判断，若是，则查询是否已经计算过模拟结果
            3. 若数据库中已存在该个体的模拟结果，直接从数据库中获取
            4. 若数据库中不存在该个体的模拟结果，调用 damperProcessor.compute() 方法计算模拟结果
            5. 将计算结果存入数据库 shuimuBIM.sqlite3 中，如果 is_storing 为 False 或者已存在该个体的模拟结果，则不进行存储操作
            6. 返回该个体的模拟结果

        Args:
            X (dict): 个体的决策变量

        Returns:
            result (dict):
                该个体的模拟结果，key 为：RM1_SUPP_VOL、RM2_SUPP_VOL...、TOT_SUPP_VOL，与数据库中的字段名一致
        """
        is_exist = 0  # 数据库中是否已经存在该个体的计算结果，0 表示不存在，否则表示索引 ID
        # 若数据库中已经存储了该个体的计算结果，直接从数据库中读取
        if self.is_using_existing_data:
            is_exist = self.DBM.check_existence(X, self.store_table_name)
            if is_exist:
                result = self.DBM.get_data(self.store_table_name, _id=is_exist)
            else:
                # 若数据库中不存在该个体的计算结果，直接计算 bim 模型
                result = DataProcessor.parse_bim_result(self.damperProcessor.compute(X))
        else:
            # 若不使用已经存在的数据，直接计算 bim 模型
            result = DataProcessor.parse_bim_result(self.damperProcessor.compute(X))
        if self.is_storing and not is_exist:
            # 若需要存储数据，且数据库中不存在该个体的计算结果，则将计算结果存入数据库
            self.DBM.add_data(result, table_name=self.store_table_name)
        return result

    def calReferObjV(self):
        """设定目标数参考值（理论最优值）

        用于计算或读取目标函数参考值，常常用待优化问题的理论全局最优解作为该参考值，用于后续进行评价进化结果的好坏。
        如果并不知道理论全局最优解是什么，可以不设置 calReferObjV() 函数。

        用于计算参考点（reference point）的适应度值，主要应用于多目标优化问题（MOEA）中的 NSGA-II 算法。
        NSGA-II 算法使用参考点来度量种群的逼近度，从而保证种群的分布均匀且具有多样性。

        Returns:
            np.array: 目标函数参考值组成的一维数组
        """
        ref_value = {'一更': 10,
                     '二更': 15,
                     '测试间一': 30,
                     '测试间二': 35,
                     '测试间三': 30,
                     '洁净走廊': 25,
                     '外走道': 0.0}
        order = ['一更', '二更', '测试间一', '测试间二', '测试间三', '洁净走廊', '外走道']  # 排序顺序
        ref_value = DataProcessor.dicts_to_ndarray(ref_value, order=order)  # 参考风量

        return ref_value.reshape(-1, 1)
