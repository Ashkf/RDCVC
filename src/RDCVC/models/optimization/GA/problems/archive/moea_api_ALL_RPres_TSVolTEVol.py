# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea

from CoreGA.utils.DampersProcessor import DampersProcessor  # bim 模型风阀操控接口
from CoreGA.utils.DataProcessor import DataProcessor
from utils.SQLiteManager import SQLiteManager  # 数据库接口

"""
    该 problem：
        * 决策变量：3 个风机频率的十倍 + 7 个送风阀开度 + 4 个回风阀开度 + 4 个排风阀开度
        * 目标函数：
        * 约束条件：
"""


class MyProblem(ea.Problem):  # 继承 Problem 父类
    def __init__(self, nndir='default', store_data=False, use_existing_data=False):
        """初始化

        Args:
            nndir (str, optional): 神经网络模型的路径。Defaults to 'default'.
            store_data (bool, optional): 是否存储数据至数据库。Defaults to False.
            use_existing_data (bool, optional): 是否使用数据库已有数据。Defaults to False.

        Author: KAI
        Date: 2023/06/28
        """
        # name 的一般命名规则：MainName + [决策变量类型] + [目标函数类型] + [约束类型]
        name = 'GAProblem-moea-API[ALL][RPres][TEVolTSVol]'  # 初始化 name（函数名称，可以随意设置），影响后续结果的保存
        M = 6  # 初始化 M（目标维数、f 的数量）
        maxormins = [1] * M  # 初始化 maxormins（目标最小最大化标记列表，1：min；-1：max）
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

        存在两个主要功能：
            1. 计算目标函数值
            2. 计算约束函数值

        Args:
            pop (Population): 一个种群对象，里面包含了种群的所有信息
        """
        Vars = pop.Phen  # 得到决策变量矩阵，这里的 Vars 是一个二维矩阵，每一行对应一个个体，每一列对应一个决策变量
        # ===================== 以下是计算目标函数值 =====================
        room_pres_ref = np.array([[10], [15], [30], [35], [30], [25], [0]])  # 房间压差设计值
        ref_room_supply_volume = np.array([235, 331, 689, 815, 722, 968])  # 设计房间送风量

        total_supply_volume = np.zeros((Vars.shape[0], 1))  # 总送风量初始化为全 0 矩阵
        total_exhaust_volume = np.zeros((Vars.shape[0], 1))  # 总排风量初始化为全 0 矩阵
        # c_room_pres = np.zeros((Vars.shape[0], 1))  # 房间压差约束函数值初始化为全 0 矩阵
        pop.ObjV = np.zeros((Vars.shape[0], self.M))  # 初始化种群的目标函数值矩阵
        for i in range(Vars.shape[0]):
            """该循环依照行对种群中的每个个体进行处理，得到每个个体的目标函数值"""
            # X 是一个个体字典，键是决策变量的名称，值是决策变量的值
            X = DataProcessor.Vars2Dict(Vars[i, :])
            # 设定固定参数
            # X['AHU_FREQ'] = 27.2
            # X['MAU_FREQ'] = 20.7
            # X['EF_FREQ'] = 43.3

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

            total_supply_volume[i, 0] = result.get("TOT_SUPP_VOL")  # 从 result 中取出总送风量
            total_exhaust_volume[i, 0] = result.get("TOT_EXH_VOL")  # 从 result 中取出总排风量

            # 值处理，如果 room_pres 中有 nan 值，说明计算结果有问题，目标函数值设置为无穷
            if np.any(np.isnan(room_pres)) or np.any(np.isnan(total_supply_volume)):
                pop.ObjV[i, 0] = np.inf  # np.inf 表示正无穷，-np.inf 表示负无穷，具体结合 problem 定义选取
                print(f'第 {(i + 1)} 个个体存在错误，目标函数值设置为 np.inf')
                continue

            # f1 = X['AHU_FREQ'] + X['MAU_FREQ'] + X['EF_FREQ']  # 目标函数值：风机频率
            # f2 = np.sqrt(MSE(room_pres, room_pres_ref))  # 目标函数值：房间压差偏差
            f_RPres1 = np.square(room_pres[0] - room_pres_ref[0])  # 房间 1 偏差
            f_RPres2 = np.square(room_pres[1] - room_pres_ref[1])  # 房间 2 偏差
            f_RPres3 = np.square(room_pres[2] - room_pres_ref[2])  # 房间 3 偏差
            f_RPres4 = np.square(room_pres[3] - room_pres_ref[3])  # 房间 4 偏差
            f_RPres5 = np.square(room_pres[4] - room_pres_ref[4])  # 房间 5 偏差
            f_RPres6 = np.square(room_pres[5] - room_pres_ref[5])  # 房间 6 偏差

            pop.ObjV[i, :] = np.hstack(
                [f_RPres1, f_RPres2, f_RPres3, f_RPres4, f_RPres5, f_RPres6])

            # c_room_pres[i, 0] = np.sqrt(MSE(room_pres, room_pres_ref)) - 10  # 目标函数值：房间压差偏差

        # ===================== 以下是计算约束 =====================
        # 采用可行性法则处理约束，numpy 的 hstack() 把列向量拼成 CV 矩阵
        # 计算约束函数值，赋值给 pop 种群对象的 CV 属性
        # 约束函数值为负表示满足约束，为正表示不满足约束，越大表示不满足约束程度越大
        # 把 c1、c2、...等列向量拼成 CV 矩阵，赋值给种群
        # -------------------------------------------------------
        ref_total_supply = 3760  # 总风量设计参考值
        ref_total_exhaust = 1600  # 总排风量设计参考值
        c_tsv = ref_total_supply - total_supply_volume  # 送风量约束，大于设计值
        c_tsv_1 = total_supply_volume - 1.1 * ref_total_supply  # 送风量约束，小于设计值 1.1 倍
        c_tev = ref_total_exhaust - total_exhaust_volume  # 排风量约束，大于设计值
        c_tev_1 = total_exhaust_volume - 1.1 * ref_total_exhaust  # 排风量约束，小于设计值 1.1 倍

        pop.CV = np.hstack([c_tsv, c_tev, c_tsv_1, c_tev_1])
        _best_id, _room_preses = DataProcessor.evaluate_room_pres(pop.ObjV)
        print(f'\033[0;33;40m'
              f'best_mean_room_pres={_room_preses[_best_id]} '
              f'|| tsv={total_supply_volume[_best_id]} '
              f'|| tev={total_exhaust_volume[_best_id]}'
              f'\033[0m')

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
