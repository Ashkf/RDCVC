"""
* Script for running the GA algorithm
*
* File: run.py
* Author: Fan Kai
* Soochow University
* Created: 2024-01-07 10:30:07
* ----------------------------
* Modified: 2024-07-07 10:03:53
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import json
import sys

import geatpy as ea
import matplotlib
from geatpy import Population

matplotlib.use("webagg")
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from MyProblem import MNSGA31
from numpy import ndarray
from rich import print
from rich.rule import Rule
from skcriteria import mkdm
from skcriteria.preprocessing.weighters import CRITIC

from src.RDCVC.models.optimization.utils.utils import gen_save_dir

sys.path.append(r"/workspace/src/")  # 因为加载完整模型需要 RDCVC 包

from pydantic import BaseModel


class OptimizationArgs(BaseModel):
    nind: int = 100
    maxgen: int = 100
    maxtime: int | None  # unit: seconds
    nnpath: str
    scaler_path: list[str]


def parse_arguments() -> OptimizationArgs:
    parser = ArgumentParser(description="Run the GA optimization algorithm.")
    parser.add_argument("--nind", type=int, help="Population size.")
    parser.add_argument(
        "--maxgen", type=int, help="Maximum number of generations."
    )
    parser.add_argument(
        "--maxtime", type=int, help="Maximum running time in seconds."
    )
    parser.add_argument(
        "--nnpath", type=str, help="Path to the neural network model."
    )
    parser.add_argument("--scaler_path", type=str, nargs=2)
    return OptimizationArgs(**vars(parser.parse_args()))


def main(args: OptimizationArgs):
    nn_model = torch.load(args.nnpath)
    problem = MNSGA31(nn_model, args.scaler_path)
    algorithm = ea.moea_NSGA2_templet(
        problem,
        Population(Encoding="RI", NIND=args.nind),
        MAXGEN=args.maxgen,
        MAXTIME=args.maxtime,
        logTras=10,  # 设置每隔多少代记录日志，若设置成 0 则表示不记录日志
        aimFuncTrace=True,  # 设置是否记录目标函数值的变化)
    )
    # algorithm.outFunc = None  # 设置每次进化记录的输出函数

    save_dirName = gen_save_dir(
        problem.name,
        algorithm.name,
        prefix="test",
        NIND=args.nind,
        MAXGEN=args.maxgen,
        MAXTIME=args.maxtime,
    )

    print("Start optimization...")
    res = ea.optimize(
        algorithm,
        seed=42,  # 随机数种子
        prophet=None,  # 先验知识
        drawLog=False,
        drawing=0,
        outputMsg=False,
        verbose=True,  # 设置是否打印输出日志信息
        saveFlag=True,
        dirName=save_dirName,
    )

    if not res["success"]:
        print(f"Optimization failed. {res['stopMsg']}")
        return

    NDSet = res["optPop"]
    print("用时：%f秒" % (res["executeTime"]))
    print("评价次数：%d次" % (res["nfev"]))
    print("非支配个体数：%d个" % (NDSet.sizes))
    print(
        f"单位时间找到帕累托前沿点个数：{NDSet.sizes / res['executeTime']: .2f} 个/s"
    )

    draw_pareto_front(NDSet, save_dirName)

    tradeoff_solution: Population = select_tradeoff_solution(
        NDSet, [min, max, min], ["E", "DPerr", "ACHerr"]
    )
    tfs_metrics = evaluate_soulution(
        tradeoff_solution, problem._model_eval, print_metrics=True
    )

    # 保存权衡解
    with open(save_dirName + "/" + "Trade_Off_Result.json", "w") as f:
        json.dump(tfs_metrics, f, indent=4)


def draw_pareto_front(NDSet: Population, save_dirName: str):
    # 使用 matplot 绘制 NDSet Pareto 前沿图
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.grid(True)
    ax.scatter(
        NDSet.ObjV[:, 0], NDSet.ObjV[:, 1], NDSet.ObjV[:, 2], c="r", marker="o"
    )
    ax.set_xlabel("Obj_E (Hz)")
    ax.set_ylabel("Obj_P (Pa)")
    ax.set_zlabel("Obj_ACH")
    plt.savefig(save_dirName + "/" + "ParetoFront.png")
    plt.show()


def evaluate_soulution(
    solution: Population, eval_func, print_metrics=False
) -> dict:
    assert solution.ChromNum == 1, "solution should be a single individual."

    ref_pres = np.array([10, 15, 32, 32, 30, 25])  # 房间压差设计值
    ref_total_SV = 3760  # 总送风量设计参考值
    ref_total_EV = 1600  #

    # ----------------------- variables ---------------------- #
    features = np.squeeze(eval_func(solution.Phen))
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

    if print_metrics:
        print("[bold reverse]Metrics:[/bold reverse]")
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


def select_tradeoff_solution(
    NDSet: Population, objectives, criteria
) -> Population:
    # ----------------- 使用客观赋权法 CRITIC 计算权衡解 ----------------- #
    if NDSet.ObjV is None:
        raise ValueError("NDSet.ObjV is None.")

    rawData_mat: ndarray = (
        NDSet.ObjV
    )  # 2D array-like, shape (n_samples, n_criteria)
    # 无量纲化处理，并使得所有指标都是最大化标准
    stdData_mat = dimensionless(rawData_mat, objectives)
    dm = mkdm(
        stdData_mat,
        [
            max,
            max,
            max,
        ],  # CRITIC 原始论文仅建议用于最大化标准 [Diakoulaki et al., 1995]
        criteria=criteria,
    )
    dm = CRITIC().transform(dm)  # 计算权重（已验证）
    weighted_scores = np.dot(stdData_mat, dm.weights)
    ascending_indices = np.argsort(weighted_scores)[::-1]
    tradeoff_solution = NDSet[ascending_indices[0]]

    return tradeoff_solution


def dimensionless(x, objectives):
    # 无量纲化处理（正/逆向化）
    # 若该指标越大越好（正向指标），则 x' = (x - x_min) / (x_max - x_min)
    # 若该指标越小越好（负向指标），则 x' = (x_max - x) / (x_max - x_min)

    # x 应当为 2D array-like, shape (n_samples, n_criteria)
    if x.shape[1] != len(objectives):
        raise ValueError("dimensionless. dimension mismatch.")
    dimless_x = x.copy()
    for i, obj in enumerate(objectives):
        if obj == min:
            _min_e = np.min(x[:, i])
            _max_e = np.max(x[:, i])
            _range = _max_e - _min_e
            dimless_x[:, i] = (_max_e - dimless_x[:, i]) / _range
        elif obj == max:
            _min_e = np.min(x[:, i])
            _max_e = np.max(x[:, i])
            _range = _max_e - _min_e
            dimless_x[:, i] = (dimless_x[:, i] - _min_e) / _range
        else:
            raise ValueError("dimensionless. invalid objective.")

    return dimless_x


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
