"""
* 用于处理数据的工具类
*
* File: DataProcessor.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-15 22:25:52
* ----------------------------
* Modified: 2024-01-06 19:50:53
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import csv
import datetime
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from utils.Ploter import Ploter  # 导入画图工具


class DataProcessor:
    @staticmethod
    def parse_bim_result(bim_result: dict):
        """解析 BIM 模型的结果

        将 BIM 插件计算结果转换成符合数据库字段格式的字典

        Args:
            bim_result (dict): BIM 插件计算结果

        Returns:
            dict: 符合数据库字段格式的字典
        """
        try:
            result_dict = {
                "DATA_SRC": "shuimuBIM",
                # 机组频率
                "MAU_FREQ": bim_result["fanConfigSnapShot"]["actualNewFrequency"],
                "AHU_FREQ": bim_result["fanConfigSnapShot"]["actualSupplyFrequency"],
                "EF_FREQ": bim_result["fanConfigSnapShot"]["actualExhaustFrequency"],
                # 送风阀角度
                "RM1_SUPP_DMPR_0": bim_result["roomNameToSupplyDamperConfigs"]["一更"][
                    0
                ]["angle"],
                "RM2_SUPP_DMPR_0": bim_result["roomNameToSupplyDamperConfigs"]["二更"][
                    0
                ]["angle"],
                "RM3_SUPP_DMPR_0": bim_result["roomNameToSupplyDamperConfigs"][
                    "测试间一"
                ][0]["angle"],
                "RM4_SUPP_DMPR_0": bim_result["roomNameToSupplyDamperConfigs"][
                    "测试间二"
                ][0]["angle"],
                "RM5_SUPP_DMPR_0": bim_result["roomNameToSupplyDamperConfigs"][
                    "测试间三"
                ][0]["angle"],
                "RM6_SUPP_DMPR_0": bim_result["roomNameToSupplyDamperConfigs"][
                    "洁净走廊"
                ][0]["angle"],
                "RM6_SUPP_DMPR_1": bim_result["roomNameToSupplyDamperConfigs"][
                    "洁净走廊"
                ][1]["angle"],
                # 回风阀角度
                "RM2_RET_DMPR_0": bim_result["roomNameToReturnDamperConfigs"]["二更"][
                    0
                ]["angle"],
                "RM3_RET_DMPR_0": bim_result["roomNameToReturnDamperConfigs"][
                    "测试间一"
                ][0]["angle"],
                "RM4_RET_DMPR_0": bim_result["roomNameToReturnDamperConfigs"][
                    "测试间二"
                ][0]["angle"],
                "RM6_RET_DMPR_0": bim_result["roomNameToReturnDamperConfigs"][
                    "洁净走廊"
                ][0]["angle"],
                # 排风阀角度
                "RM3_EXH_DMPR_0": bim_result["roomNameToExhaustDamperConfigs"][
                    "测试间一"
                ][0]["angle"],
                "RM4_EXH_DMPR_0": bim_result["roomNameToExhaustDamperConfigs"][
                    "测试间二"
                ][0]["angle"],
                "RM5_EXH_DMPR_0": bim_result["roomNameToExhaustDamperConfigs"][
                    "测试间三"
                ][0]["angle"],
                "RM5_EXH_DMPR_1": bim_result["roomNameToExhaustDamperConfigs"][
                    "测试间三"
                ][1]["angle"],
                # 房间 1（一更）状态
                "RM1_DIFF_VOL": bim_result["roomNameToRemainWindVolume"]["一更"],
                "RM1_EXH_VOL": bim_result["roomNameToExhaustWindVolume"]["一更"],
                "RM1_PRES": bim_result["roomNameToRealPressure"]["一更"],
                "RM1_RET_VOL": bim_result["roomNameToReturnWindVolume"]["一更"],
                "RM1_SUPP_VOL": bim_result["roomNameToSupplyWindVolume"]["一更"],
                # 房间 2（二更）状态
                "RM2_DIFF_VOL": bim_result["roomNameToRemainWindVolume"]["二更"],
                "RM2_EXH_VOL": bim_result["roomNameToExhaustWindVolume"]["二更"],
                "RM2_PRES": bim_result["roomNameToRealPressure"]["二更"],
                "RM2_RET_VOL": bim_result["roomNameToReturnWindVolume"]["二更"],
                "RM2_SUPP_VOL": bim_result["roomNameToSupplyWindVolume"]["二更"],
                # 房间 3（测试间一）状态
                "RM3_DIFF_VOL": bim_result["roomNameToRemainWindVolume"]["测试间一"],
                "RM3_EXH_VOL": bim_result["roomNameToExhaustWindVolume"]["测试间一"],
                "RM3_PRES": bim_result["roomNameToRealPressure"]["测试间一"],
                "RM3_RET_VOL": bim_result["roomNameToReturnWindVolume"]["测试间一"],
                "RM3_SUPP_VOL": bim_result["roomNameToSupplyWindVolume"]["测试间一"],
                # 房间 4（测试间二）状态
                "RM4_DIFF_VOL": bim_result["roomNameToRemainWindVolume"]["测试间二"],
                "RM4_EXH_VOL": bim_result["roomNameToExhaustWindVolume"]["测试间二"],
                "RM4_PRES": bim_result["roomNameToRealPressure"]["测试间二"],
                "RM4_RET_VOL": bim_result["roomNameToReturnWindVolume"]["测试间二"],
                "RM4_SUPP_VOL": bim_result["roomNameToSupplyWindVolume"]["测试间二"],
                # 房间 5（测试间三）状态
                "RM5_DIFF_VOL": bim_result["roomNameToRemainWindVolume"]["测试间三"],
                "RM5_EXH_VOL": bim_result["roomNameToExhaustWindVolume"]["测试间三"],
                "RM5_PRES": bim_result["roomNameToRealPressure"]["测试间三"],
                "RM5_RET_VOL": bim_result["roomNameToReturnWindVolume"]["测试间三"],
                "RM5_SUPP_VOL": bim_result["roomNameToSupplyWindVolume"]["测试间三"],
                # 房间 6（洁净走廊）状态
                "RM6_DIFF_VOL": bim_result["roomNameToRemainWindVolume"]["洁净走廊"],
                "RM6_EXH_VOL": bim_result["roomNameToExhaustWindVolume"]["洁净走廊"],
                "RM6_PRES": bim_result["roomNameToRealPressure"]["洁净走廊"],
                "RM6_RET_VOL": bim_result["roomNameToReturnWindVolume"]["洁净走廊"],
                "RM6_SUPP_VOL": bim_result["roomNameToSupplyWindVolume"]["洁净走廊"],
                # 房间 7（洁净走廊）状态
                "RM7_DIFF_VOL": bim_result["roomNameToRemainWindVolume"]["外走道"],
                "RM7_EXH_VOL": bim_result["roomNameToExhaustWindVolume"]["外走道"],
                "RM7_PRES": bim_result["roomNameToRealPressure"]["外走道"],
                "RM7_RET_VOL": bim_result["roomNameToReturnWindVolume"]["外走道"],
                "RM7_SUPP_VOL": bim_result["roomNameToSupplyWindVolume"]["外走道"],
                # 系统状态
                "TOT_EXH_VOL": bim_result["exhaustQe"],
                "TOT_FRSH_VOL": bim_result["newWindVolumeQf"],
                "TOT_RET_VOL": bim_result["totalNewWindVolumeQr"],
                "TOT_SUPP_VOL": bim_result["totalWindVolumeQs"],
            }
            return result_dict
        except TypeError as e:
            print(
                f"parse_bim_result 时出错。错误信息：{e}。"
                f"时间：{datetime.datetime.now()}。"
            )
            print("主动退出程序。")
            sys.exit(1)

    @staticmethod
    def cal_proportions(arr):
        """计算数组中各个元素在总和中所占的比例

        Args:
            arr (np.array): 一个数组

        Returns:
            np.array: 数组中的每个元素代表 arr 中对应元素在总和中所占的比例
        """
        # 计算数组中所有元素的总和
        total_sum = np.sum(arr)

        # 对每个元素进行除法运算，得到该元素在总和中所占的比例
        proportions = arr / total_sum

        return proportions

    @staticmethod
    def dicts_to_ndarray(*dicts, order):
        """将多个同 keys 字典，按照 order 列表的顺序转换成一个 ndarry

        dicts 的 keys 必须相同，否则会报错

        Examples:
            dict1 = {'a': 1, 'b': 2, 'c': 3}
            dict2 = {'a': 4, 'b': 5, 'c': 6}
            dict3 = {'a': 7, 'b': 8, 'c': 9}
            order = ['a', 'b', 'c']
            dicts_to_ndarray(dict1, dict2, dict3, order)
            ==>array([[1, 4, 7],
                        [2, 5, 8],
                        [3, 6, 9]])

        Args:
            *dicts (dict): 字典，可以传入单个或多个字典
            order (list): 排序顺序

        Returns:
            arr (np.ndarray): 一个 ndarry
        """
        # --------------------- 检查 keys 是否相同 --------------------- #
        keys_set = set(dicts[0].keys())
        if not all(set(d.keys()) == keys_set for d in dicts[1:]):
            raise ValueError("dicts 的 keys 必须相同")

        # -------------------------- 排序 -------------------------- #
        sorted_keys = sorted(order, key=lambda x: order.index(x))
        arr = np.array([[d[k] for d in dicts] for k in sorted_keys])
        return arr

    @staticmethod
    def dampers_result_to_dict(result, systemT):
        """将通过 api 同步计算得到的结果中，解析得到的 damper 配置信息转换成 dampers 字典

        dict 的结构如下：
        {
            damper_uid:{
                "id": damper_id,
                "angle": damper_angle,
                "room": room_name,
                "systemT": "supply"/"return"/"exhaust",
                "name": None,
                "elementId": damper_elementId
                }
        }

        Args:
            result (dict): 从数据库中查询到的 damper 配置信息
            systemT (str): 空调系统类型，"supply"/"return"/"exhaust"

        Returns:
            dampers (dict): 转换后的 dampers 字典
        """
        dampers = {}
        for room_name, damper_configs in result.items():
            for damper_config in damper_configs:
                dampers[damper_config["uniqueId"]] = {
                    "id": damper_config["id"],
                    "angle": damper_config["angle"],
                    "room": room_name,
                    "systemT": systemT,
                    "name": None,
                    "elementId": damper_config["elementId"],
                }

        return dampers

    @staticmethod
    def Vars2Dict(Vars):
        X = {
            "MAU_FREQ": Vars[0] / 10,
            "AHU_FREQ": Vars[1] / 10,
            "EF_FREQ": Vars[2] / 10,
            "RM1_SUPP_DMPR_0": Vars[3],
            "RM2_SUPP_DMPR_0": Vars[4],
            "RM3_SUPP_DMPR_0": Vars[5],
            "RM4_SUPP_DMPR_0": Vars[6],
            "RM5_SUPP_DMPR_0": Vars[7],
            "RM6_SUPP_DMPR_0": Vars[8],
            "RM6_SUPP_DMPR_1": Vars[9],
            "RM2_RET_DMPR_0": Vars[10],
            "RM3_RET_DMPR_0": Vars[11],
            "RM4_RET_DMPR_0": Vars[12],
            "RM6_RET_DMPR_0": Vars[13],
            "RM3_EXH_DMPR_0": Vars[14],
            "RM4_EXH_DMPR_0": Vars[15],
            "RM5_EXH_DMPR_0": Vars[16],
            "RM5_EXH_DMPR_1": Vars[17],
        }
        return X

    @staticmethod
    def X2Tensor(X):
        """将 X 转换成 Tensor

        Args:
            X (dict): X 字典
        """
        X_tensor = torch.zeros(1, 18)
        X_tensor[0, 0] = X["MAU_FREQ"]
        X_tensor[0, 1] = X["AHU_FREQ"]
        X_tensor[0, 2] = X["EF_FREQ"]
        X_tensor[0, 3] = X["RM1_SUPP_DMPR_0"]
        X_tensor[0, 4] = X["RM2_SUPP_DMPR_0"]
        X_tensor[0, 5] = X["RM3_SUPP_DMPR_0"]
        X_tensor[0, 6] = X["RM4_SUPP_DMPR_0"]
        X_tensor[0, 7] = X["RM5_SUPP_DMPR_0"]
        X_tensor[0, 8] = X["RM6_SUPP_DMPR_0"]
        X_tensor[0, 9] = X["RM6_SUPP_DMPR_1"]
        X_tensor[0, 10] = X["RM2_RET_DMPR_0"]
        X_tensor[0, 11] = X["RM3_RET_DMPR_0"]
        X_tensor[0, 12] = X["RM4_RET_DMPR_0"]
        X_tensor[0, 13] = X["RM6_RET_DMPR_0"]
        X_tensor[0, 14] = X["RM3_EXH_DMPR_0"]
        X_tensor[0, 15] = X["RM4_EXH_DMPR_0"]
        X_tensor[0, 16] = X["RM5_EXH_DMPR_0"]
        X_tensor[0, 17] = X["RM5_EXH_DMPR_1"]
        return X_tensor

    @staticmethod
    def process_result_nsga2(result, algorithm, problem, from_api=True):
        best_Vars = result["optPop"].Phen
        best_ObjVs = result["optPop"].ObjV
        # 将优化结果保存到文件
        # optPop.Phen 和 optPop.ObjV 中元素，依据行索引对应，保存到同一个 csv 文件中
        with open(algorithm.dirName + "/" + "optPop.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "Phen", "ObjV"])
            for i in range(best_Vars.shape[0]):
                writer.writerow([i, best_Vars[i], best_ObjVs[i]])
            print(f"optPop.csv saved to {algorithm.dirName}")

        # 从 optPop 中获取最优解
        # 1. 依据 ObjV 的第二列和第三列，筛选出符合条件的行，保存到 criterion
        # 2. 依据 criterion 的第一列，选取符合条件的最小值，保存到 best_index
        # criterion = best_ObjVs.copy()
        # criterion[:, 1] = np.where(criterion[:, 1] > 20, 0, 1)  # 过滤压强均差大于 20
        # criterion[:, 2] = np.where(criterion[:, 2] > 100, 0, 1)  # 过滤送风量均差大于 100
        # # criterion 第二列和第三列均不为零的行中，选取第一列最小的行，将行索引赋值给 best_index
        # criterion = criterion[:, 0] * criterion[:, 1] * criterion[:, 2]  # 将第一列、第二列和第三列相乘
        # criterion = np.where(criterion == 0, 99999, criterion)  # 将 0 替换为 99999
        # best_index = criterion.argmin()  # 将 0 替换为 99999，再选取最小值的索引
        # print(f'best index: {best_index}' if criterion[
        #                                          best_index] != 99999 else '!!!!! 没有压强误差小于 20 且送风量小于 100 的解!!!!!')

        # # 从 optPop 中获取最优解
        # # 1. 依据 ObjV 的 [:, 3:8], 求房间误差均值，保存到 criterion 第二列
        # # 2. 依据 ObjV 的 [:, 0:3], 求风机频率均值，保存到 criterion 第一列
        # # 3. 依据 criterion，选取符合条件的最小值，保存到 best_index
        # criterion = np.sum(best_ObjVs[:, 3:9], axis=1) / 6  # 将第 3-8 列相加，求房间误差均值
        # criterion = np.where(criterion > 10, 999999, criterion)  # 过滤压强均差大于 10
        # criterion += np.sum(best_ObjVs[:, 0:3], axis=1) / 3  # 将第 0-2 列相加，求风机频率均值
        # best_index = criterion.argmin()  # 选取风机频率最小值的索引
        # print(f'best index: {best_index}'
        #       if criterion[best_index] < 99999 else '!!!!! 没有压强误差小于 10 的解!!!!!')

        # 从 optPop 中获取最优解
        # 1. 依据 ObjV 的 [:, :], 求房间压差误差均值，保存到 criterion 第一列
        # 2. 依据 criterion，选取符合条件的最小值，保存到 best_index
        best_index, criterion = DataProcessor.evaluate_room_pres(best_ObjVs)

        # 输出最佳个体的决策变量和目标函数值，并保存为 criterion_best_chrom.txt
        best_Phen = best_Vars[best_index]
        with open(algorithm.dirName + "/" + "criterion_best_chrom.txt", "w") as f:
            print(f"best index: {best_index}")
            f.write("best index: " + str(best_index) + "\n")

            print(f"criterion: {criterion[best_index]}")
            f.write("criterion: " + str(criterion[best_index]) + "\n")

            print(f"best Phen: {best_Vars[best_index]}")
            f.write("best Phen: " + str(best_Vars[best_index]) + "\n")

            print(f"best ObjV: {best_ObjVs[best_index]}")
            f.write("best ObjV: " + str(best_ObjVs[best_index]) + "\n")

        # =====
        # 最优解
        # =====
        X_opt = DataProcessor.Vars2Dict(best_Phen.reshape(-1))  # 优化后的决策变量字典
        opt_result = problem.get_result(X_opt)  # 优化后的结果
        with open(algorithm.dirName + "/" + "opt_result.txt", "w") as f:
            f.write(str(opt_result))

    @staticmethod
    def evaluate_room_pres(room_data):
        """评估房间压差误差均值

        Args:
            room_data (np.ndarray): 房间压差数据，shape=(pop_size, 6)
        """
        room_pres_err = np.sqrt(room_data)  # 各房间压差误差
        criterion = np.sum(room_pres_err, axis=1) / 6  # 房间压差误差均值
        best_index = criterion.argmin()  # 选取风机频率最小值的索引
        return best_index, criterion

    @staticmethod
    def process_result_soea(result, algorithm, problem, from_api=True):
        # =======================
        # 进化过程的种群信息 (trace)
        # =======================
        # algorithm.trace 的结构：{'f_best': [], 'f_avg': []}
        Ploter.plot_trace(
            algorithm.trace, save_path=algorithm.dirName + "/" + "trace.png"
        )
        # 保存 trace 原始数据
        with open(algorithm.dirName + "/" + "trace.txt", "w") as file:
            file.write(str(algorithm.trace))
        # =====
        # 最优解
        # =====
        X_opt = DataProcessor.Vars2Dict(
            result["Vars"].reshape(-1)
        )  # 优化后的决策变量字典
        opt_result = problem.get_result(X_opt)  # 优化后的结果
        opt_pressure = np.zeros((7, 1))
        if from_api:
            #  从 result 中取出取出房间压差
            room_pres_keys = [
                "RM1_PRES",
                "RM2_PRES",
                "RM3_PRES",
                "RM4_PRES",
                "RM5_PRES",
                "RM6_PRES",
                "RM7_PRES",
            ]
            opt_pressure = np.array(
                [opt_result.get(key) for key in room_pres_keys]
            ).reshape(-1, 1)
        opt_pres_rmse = np.sqrt(MSE(opt_pressure, problem.calReferObjV()))
        opt_freq = [
            X_opt["MAU_FREQ"],
            X_opt["AHU_FREQ"],
            X_opt["EF_FREQ"],
        ]  # 优化后的风机频率

        # 压差状态图
        res_data = {
            "ref_press": problem.calReferObjV(),
            "opt_press": opt_pressure,
        }  # 绘图数据
        Ploter.plot_pres_per_rooms(
            res_data, save_path=algorithm.dirName + "/" + "result.png"
        )
        plt.show()

        # ================== 输出状态 ==================
        print(f'求解状态：{result["success"]}')
        print(f'评价次数：{result["nfev"]}')
        print(f'时间花费：{result["executeTime"]}秒')
        if result["success"]:
            print(f'最优的目标函数值为：{result["ObjV"]}')
            print(f'最优的决策变量值为：{result["Vars"]}')
            print(f"最优的风机频率为：{opt_freq}")
            print(f"最优的压差为：{opt_pres_rmse}")
        else:
            print("此次未找到可行解。")

    @staticmethod
    def process_result_moea(result, algorithm, problem):
        """处理 Geatpy 多目标优化的结果"""

        # =========== 进化过程的种群信息 (trace) ===========
        # =========== 最优解 ===========
        # X_opt = DataProcessor.Vars2Dict(result['Vars'].reshape(-1))  # 优化后的决策变量字典
        # opt_result = problem.get_result(X_opt)  # 优化后的结果
        # opt_pressure = opt_result[:, :-1].reshape(-1, 1)  # 优化后的压差
        # opt_pres_rmse = np.sqrt(MSE(opt_pressure, problem.calReferObjV()))
        # opt_freq = [X_opt['MAU_FREQ'], X_opt['AHU_FREQ'], X_opt['EF_FREQ']]  # 优化后的风机频率
        #
        # # 压差状态图
        # res_data = {'ref_press': problem.calReferObjV(), 'opt_press': opt_pressure}  # 绘图数据
        # Ploter.plot_pres_per_rooms(res_data, save_path=algorithm.dirName + '/' + 'result.png')
        # plt.show()
        #
        # # ================== 输出状态 ==================
        # print(f'求解状态：{result["success"]}')
        # print(f'评价次数：{result["nfev"]}')
        # print(f'时间花费：{result["executeTime"]}秒')
        # if result['success']:
        #     print(f'最优的目标函数值为：{result["ObjV"]}')
        #     print(f'最优的决策变量值为：{result["Vars"]}')
        #     print(f'最优的风机频率为：{opt_freq}')
        #     print(f'最优的压差为：{opt_pres_rmse}')
        # else:
        #     print('此次未找到可行解。')
