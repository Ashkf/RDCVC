"""
* Script for running the GA algorithm
*
* File: run.py
* Author: Fan Kai
* Soochow University
* Created: 2024-01-07 10:30:07
* ----------------------------
* Modified: 2024-07-05 12:19:37
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import json
import os
import sys
from datetime import datetime

import geatpy as ea
import numpy as np
import pandas as pd
import torch
from MyProblem import SN1
from pop_history import pop_history
from rich import print
from rich.rule import Rule

sys.path.append(r"/workspace/src/")  # 因为加载完整模型需要 RDCVC 包


def results_postprocess(res, eval_func):
    """
    Args:
        res (dict): 一个保存着结果的字典。内容为：
            {
                'success': True or False,  # 表示算法是否成功求解。
                'stopMsg': xxx,  # 存储着算法停止原因的字符串。
                'optPop': xxx,  # 存储着算法求解结果的种群对象。如果无可行解，则 optPop.sizes=0。optPop.Phen 为决策变量矩阵，optPop.ObjV 为目标函数值矩阵。
                'lastPop': xxx,  # 算法进化结束后的最后一代种群对象。
                'Vars': xxx,  # 等于 optPop.Phen，此处即最优解。若无可行解，则 Vars=None。
                'ObjV': xxx,  # 等于 optPop.ObjV，此处即最优解对应的目标函数值。若无可行解，ObjV=None。
                'CV': xxx,  # 等于 optPop.CV，此处即最优解对应的违反约束程度矩阵。若无可行解，CV=None。
                'startTime': xxx,  # 程序执行开始时间。
                'endTime': xxx,  # 程序执行结束时间。
                'executeTime': xxx,  # 算法所用时间。
                'nfev': xxx,  # 算法评价次数
                'gd': xxx,  # (多目标优化且给定了理论最优解时才有) GD 指标值。
                'igd': xxx,  # (多目标优化且给定了理论最优解时才有) IGD 指标值。
                'hv': xxx,  # (多目标优化才有) HV 指标值。
                'spacing': xxx  # (多目标优化才有) Spacing 指标值。
            }
    """
    if not res["success"]:
        print(f"Optimization failed. {res['stopMsg']}")
        return

    ref_pres = np.array([10, 15, 32, 32, 30, 25])  # 房间压差设计值
    ref_total_SV = 3760  # 总送风量设计参考值
    ref_total_EV = 1600  # 总排风量设计参考值
    # ----------------------- variables ---------------------- #
    features = np.squeeze(eval_func(res["Vars"]))
    ctrlable_vars = {
        "MAU_FREQ": features[0],
        "AHU_FREQ": features[1],
        "EF_FREQ": features[2],
        "RM1_SUPP_DMPR_0": features[3],
        "RM2_SUPP_DMPR_0": features[4],
        "RM3_SUPP_DMPR_0": features[5],
        "RM4_SUPP_DMPR_0": features[6],
        "RM5_SUPP_DMPR_0": features[7],
        "RM6_SUPP_DMPR_0": features[8],
        "RM6_SUPP_DMPR_1": features[9],
        "RM2_RET_DMPR_0": features[10],
        "RM3_RET_DMPR_0": features[11],
        "RM4_RET_DMPR_0": features[12],
        "RM6_RET_DMPR_0": features[13],
        "RM3_EXH_DMPR_0": features[14],
        "RM4_EXH_DMPR_0": features[15],
        "RM5_EXH_DMPR_0": features[16],
        "RM5_EXH_DMPR_1": features[17],
    }
    ctrled_vars = {
        "TOT_FRSH_VOL": features[18],
        "TOT_SUPP_VOL": features[19],
        "TOT_EXH_VOL": features[20],
        "TOT_RET_VOL": features[21],
        "RM1_PRES": features[22],
        "RM2_PRES": features[23],
        "RM3_PRES": features[24],
        "RM4_PRES": features[25],
        "RM5_PRES": features[26],
        "RM6_PRES": features[27],
    }

    # ------------------------ metrics ----------------------- #
    metrics = {}
    # 风机平均频率
    metrics["freq_mean"] = np.mean(features[:3])

    # 风量偏差
    metrics["SV_Err"] = ctrled_vars["TOT_SUPP_VOL"] - ref_total_SV
    metrics["EV_Err"] = ctrled_vars["TOT_EXH_VOL"] - ref_total_EV

    # 房间压差状况
    # RMSE
    room_pres = features[-6:]
    metrics["rm_pres_RMSE"] = np.sqrt(np.mean(np.square(room_pres - ref_pres)))

    # maximum error
    metrics["rm_pres_MaxErr"] = np.max(np.abs(room_pres - ref_pres))

    # differential pressure gradient
    # RM1 ~ RM6 PRES 大小关系应当符合：RM1 < RM2 < RM6, RM6 < RM3, RM6 < RM4, RM6 < RM5
    room_pres_ordered = (
        (room_pres[0] < room_pres[1] < room_pres[5])  # 一更 < 二更 < 洁净走廊
        and (room_pres[5] < room_pres[2])  # 洁净走廊 < 测试间一
        and (room_pres[5] < room_pres[3])  # 洁净走廊 < 测试间二
        and (room_pres[5] < room_pres[4])  # 洁净走廊 < 测试间三
    )
    if not room_pres_ordered:
        print(
            "DP gradient Warning: 一更 > 二更 or 二更 > 洁净走廊 or 洁净走廊 > 测试间一 or"
            " 洁净走廊 > 测试间二 or 洁净走廊 > 测试间三。"
        )
    metrics["rm_pres_Ordered"] = bool(room_pres_ordered)

    # ------------------------ output ------------------------ #
    print("[bold reverse]Best results:[/bold reverse]")
    print(Rule())
    print(pd.DataFrame(metrics, index=[0]))
    print(Rule())
    print(pd.DataFrame(ctrlable_vars, index=[0]))
    print(Rule())
    print(pd.DataFrame(ctrled_vars, index=[0]))

    return {
        **{"ctrlable_vars": ctrlable_vars},
        **{"ctrled_vars": ctrled_vars},
        **{"metrics": metrics},
    }


def _generate_archive_directory(
    problem_name, algorithm_name, model_type, NIND, MAXGEN, MAXTIME
):
    """生成 GA 运行存档目录

    Args:
        problem_name (str): 问题名称
        algorithm_name (str): 算法名称
        model_type (str): 模型类型
        NIND (int): 种群规模
        MAXGEN (int): 最大进化代数
        MAXTIME (int): 最大运行时间 (s)

    Returns:
        str: 存档目录
    """
    _base_directory = os.path.join(os.getcwd(), "checkpoints/GA")

    # ------------------------- 存档文件名 ------------------------ #
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%dT%H%M%S")
    dir_name = f"{problem_name}_{algorithm_name}_{model_type}_N{NIND}_G{MAXGEN}_T{MAXTIME}_{timestamp}"

    # ------------------------ 构建存档目录 ------------------------ #
    archive_directory = os.path.join(_base_directory, dir_name).replace("\\", "/")
    os.makedirs(archive_directory, exist_ok=False)

    return archive_directory


def main(args):
    # ---------------------- Preprocess ---------------------- #
    model = torch.load(args["nnpath"])

    # ----------------------- algorithm ---------------------- #
    problem = SN1(model, args["scaler_path"])

    population = ea.Population(Encoding="RI", NIND=args["nind"])

    algorithm = ea.soea_SEGA_templet(
        problem,
        population,
        MAXGEN=args["maxgen"],
        MAXTIME=args["maxtime"],
        logTras=1,  # 设置每隔多少代记录日志，若设置成 0 则表示不记录日志
        aimFuncTrace=True,  # 设置是否记录目标函数值的变化)
    )
    # algorithm.outFunc = None  # 设置每次进化记录的输出函数

    # ----------------------- Optimize ----------------------- #
    print("Start optimization...")
    save_dirName = _generate_archive_directory(
        problem.name,
        algorithm.name,
        model_type="nn",
        NIND=args["nind"],
        MAXGEN=args["maxgen"],
        MAXTIME=args["maxtime"],
    )

    res = ea.optimize(
        algorithm,
        seed=42,  # 随机数种子
        prophet=None,  # 先验知识
        drawLog=True,
        drawing=1,
        outputMsg=True,
        verbose=True,  # 设置是否打印输出日志信息
        saveFlag=True,
        dirName=save_dirName,
    )

    # ---------------------- Postprocess --------------------- #

    # 保存进化过程的种群信息 (trace)
    # algorithm.trace 的结构：{'f_best': [], 'f_avg': []}
    pd.DataFrame(algorithm.trace).to_csv(save_dirName + "/" + "trace.csv", index=False)
    pd.DataFrame(pop_history).to_csv(save_dirName + "/" + "pop_history.csv", index=False)

    # 保存最优解
    res = results_postprocess(res, problem._model_eval)
    with open(save_dirName + "/" + "BestResults.json", "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    args = {
        "nind": 10000,  # 种群规模
        "maxgen": 2000,  # 最大进化代数
        "maxtime": 60,  # 最大运行时间 (s)
        "nnpath": "/workspace/checkpoints/cvcnet-mtl-mlp_18_1_2_3_64-64-64_64-64-64_BS16_LR0.01_EP10000_2024-02-26T15-33-04/final_model.pth",  # 神经网络模型存档路径，
        "scaler_path": [
            "/workspace/checkpoints/cvcnet-mtl-mlp_18_1_2_3_64-64-64_64-64-64_BS16_LR0.01_EP10000_2024-02-26T15-33-04/x_normalizer.pkl",
            "/workspace/checkpoints/cvcnet-mtl-mlp_18_1_2_3_64-64-64_64-64-64_BS16_LR0.01_EP10000_2024-02-26T15-33-04/y_normalizer.pkl",
        ],  # 数据标准化器存档路径
    }
    main(args)
